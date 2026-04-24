"""
Phase 3: 联合深度 + 模态选择器
==============================

联合预测:
1. 循环深度 (4/8/12/16)
2. 主要模态 (TEXT/IMAGE/TABLE/EQUATION)

基于:
- 复杂度网络 (ComplexityAwareLoopDepth)
- 内容路由网络 (ContentTypeRouter)
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.main_p0 import ComplexityAwareLoopDepth
from open_mythos.rag.content_router import ContentTypeRouter, ModalityType


# ============================================================================
# Joint Depth + Modality Selector
# ============================================================================


class JointDepthModalitySelector(nn.Module):
    """
    联合深度 + 模态选择器。

    同时预测:
    1. 循环深度 (4/8/12/16)
    2. 主要模态 (TEXT/IMAGE/TABLE/EQUATION)

    决策矩阵 (可学习):
        复杂度等级 × 模态类型 → 推荐深度

    深度选项: [4, 8, 12, 16]
    模态选项: [TEXT, IMAGE, TABLE, EQUATION]
    """

    DEPTH_OPTIONS = [4, 8, 12, 16]
    NUM_MODALITIES = 4

    def __init__(
        self,
        cfg: Any,  # MythosConfig
        complexity_weight: float = 0.5,
        modality_weight: float = 0.5,
    ):
        """
        Args:
            cfg: MythosConfig
            complexity_weight: 复杂度预测权重
            modality_weight: 模态预测权重
        """
        super().__init__()
        self.cfg = cfg
        self.complexity_weight = complexity_weight
        self.modality_weight = modality_weight

        # 复杂度评估网络
        self.complexity_net = ComplexityAwareLoopDepth(cfg)

        # 模态路由网络
        self.modality_router = ContentTypeRouter(cfg.dim)

        # 深度决策表: (num_complexity_levels, num_modalities, num_depth_options)
        # 每个组合有一个深度分布
        self.depth_table = nn.Parameter(
            torch.randn(4, self.NUM_MODALITIES, len(self.DEPTH_OPTIONS)) * 0.02
        )

        # 深度预测头
        self.depth_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 2),
            nn.GELU(),
            nn.Linear(cfg.dim // 2, len(self.DEPTH_OPTIONS)),
        )

        # 置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 4),
            nn.GELU(),
            nn.Linear(cfg.dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        use_routing: bool = True,
        force_modality: Optional[str] = None,
        force_depth: Optional[int] = None,
    ) -> dict:
        """
        联合预测深度和模态。

        Args:
            h: Hidden state (B, T, dim)
            use_routing: 是否使用路由网络
            force_modality: 强制指定模态 (用于调试)
            force_depth: 强制指定深度 (用于调试)

        Returns:
            dict with:
                - depth: 推荐的循环深度 (int)
                - modality: 主要模态 (str)
                - confidence: 置信度 (float)
                - complexity: 复杂度分数 (float)
                - depth_logits: 深度分布 logits
                - modality_probs: 模态分布
        """
        B, T, D = h.shape

        # 1. 复杂度评估
        complexity = self.complexity_net(h)  # (B,) in [0, 1]
        complexity_level = self._complexity_to_level(complexity)  # (B,) in [0, 1, 2, 3]

        # 2. 模态路由
        if use_routing and force_modality is None:
            routing_info = self.modality_router(h)
            modality_probs = routing_info["probs"]  # (B, 4)
            modality_idx = modality_probs.argmax(dim=-1)  # (B,)
        else:
            # 强制模态
            if force_modality is not None:
                mod_name = force_modality.upper()
                if mod_name == "TEXT":
                    modality_idx = torch.zeros(B, dtype=torch.long, device=h.device)
                elif mod_name == "IMAGE":
                    modality_idx = torch.ones(B, dtype=torch.long, device=h.device)
                elif mod_name == "TABLE":
                    modality_idx = torch.full((B,), 2, dtype=torch.long, device=h.device)
                elif mod_name == "EQUATION":
                    modality_idx = torch.full((B,), 3, dtype=torch.long, device=h.device)
                else:
                    modality_idx = torch.zeros(B, dtype=torch.long, device=h.device)
            else:
                modality_idx = torch.zeros(B, dtype=torch.long, device=h.device)

            routing_info = {"probs": torch.zeros(B, self.NUM_MODALITIES, device=h.device)}
            routing_info["probs"][:, modality_idx] = 1.0
            modality_probs = routing_info["probs"]

        # 3. 深度预测
        if force_depth is not None:
            # 强制深度
            depth_idx = self.DEPTH_OPTIONS.index(force_depth) if force_depth in self.DEPTH_OPTIONS else 1
            depth_logits = torch.zeros(B, len(self.DEPTH_OPTIONS), device=h.device)
            depth_logits[:, depth_idx] = 10.0
        else:
            # 联合预测: 复杂度 + 模态 → 深度
            depth_logits = self._predict_depth(complexity_level, modality_idx)  # (B, num_depth_options)

        # 采样深度 (贪婪或采样)
        depth_probs = F.softmax(depth_logits, dim=-1)
        depth_idx = depth_probs.argmax(dim=-1)  # (B,)
        depths = [self.DEPTH_OPTIONS[i] for i in depth_idx]

        # 4. 置信度
        confidence = self.confidence_head(h.mean(dim=1)).squeeze(-1)  # (B,)

        # 组装结果
        result = {
            "depth": depths[0] if B == 1 else depths,
            "modality": ["TEXT", "IMAGE", "TABLE", "EQUATION"][modality_idx[0].item()] if B == 1
                       else [ModalityType(m) for m in modality_idx],
            "confidence": confidence,
            "complexity": complexity,
            "depth_logits": depth_logits,
            "modality_probs": modality_probs,
            "depth_probs": depth_probs,
        }

        return result

    def _complexity_to_level(self, complexity: torch.Tensor) -> torch.Tensor:
        """
        将复杂度分数转换为等级 (0-3)。

        0: 简单 (complexity < 0.25)
        1: 中等 (0.25 <= complexity < 0.5)
        2: 较难 (0.5 <= complexity < 0.75)
        3: 困难 (complexity >= 0.75)
        """
        levels = torch.zeros_like(complexity, dtype=torch.long)
        levels[complexity >= 0.25] = 1
        levels[complexity >= 0.5] = 2
        levels[complexity >= 0.75] = 3
        return levels

    def _predict_depth(
        self,
        complexity_level: torch.Tensor,
        modality_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于复杂度和模态预测深度。

        Args:
            complexity_level: (B,) 复杂度等级 0-3
            modality_idx: (B,) 模态索引 0-3

        Returns:
            (B, num_depth_options) 深度 logits
        """
        B = complexity_level.shape[0]

        # 从深度表中查找
        # depth_table: (4, 4, num_depth_options)
        depth_logits = self.depth_table[complexity_level, modality_idx]  # (B, num_depth_options)

        # 加上基于 hidden state 的预测
        # (简化为直接返回查表结果)
        return depth_logits


# ============================================================================
# Adaptive Loop Controller
# ============================================================================


class AdaptiveLoopController(nn.Module):
    """
    自适应循环控制器。

    动态决定:
    1. 是否继续循环
    2. 当前循环深度
    3. 是否启用检索

    基于:
    - 循环收敛度 (hidden state 变化)
    - ACT 累积概率
    - 复杂度评估
    """

    def __init__(
        self,
        cfg: Any,
        joint_selector: Optional[JointDepthModalitySelector] = None,
        convergence_threshold: float = 0.01,
        max_loops_without_progress: int = 3,
    ):
        """
        Args:
            cfg: MythosConfig
            joint_selector: 联合选择器
            convergence_threshold: 收敛阈值
            max_loops_without_progress: 无进步最大循环数
        """
        super().__init__()
        self.cfg = cfg
        self.joint_selector = joint_selector or JointDepthModalitySelector(cfg)
        self.convergence_threshold = convergence_threshold
        self.max_loops_without_progress = max_loops_without_progress

        # 收敛度评估
        self.convergence_detector = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 2),
            nn.GELU(),
            nn.Linear(cfg.dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_prev: torch.Tensor,
        h_curr: torch.Tensor,
        loop_idx: int,
        cumulative_p: torch.Tensor,
        total_loops: int,
    ) -> dict:
        """
        评估是否继续循环。

        Args:
            h_prev: 上一步 hidden state
            h_curr: 当前 hidden state
            loop_idx: 当前循环索引
            cumulative_p: ACT 累积概率
            total_loops: 总循环数

        Returns:
            dict with:
                - should_continue: 是否继续
                - recommended_depth: 推荐深度
                - convergence_score: 收敛度
                - reason: 决策原因
        """
        device = h_curr.device
        B = h_curr.shape[0]

        # 1. 收敛度检测
        delta = torch.norm(h_curr - h_prev, dim=-1)  # (B,)
        convergence = 1.0 / (1.0 + delta / 10.0)  # 归一化
        convergence_score = self.convergence_detector(h_curr).squeeze(-1)  # (B,)

        # 2. ACT 判断
        act_halted = cumulative_p >= self.cfg.act_threshold  # (B,)

        # 3. 联合选择
        selector_output = self.joint_selector(h_curr)
        recommended_depth = selector_output["depth"]

        # 4. 综合决策
        should_continue = torch.ones(B, dtype=torch.bool, device=device)
        reasons = ["continue"] * B

        for b in range(B):
            # ACT 已停止
            if act_halted[b]:
                should_continue[b] = False
                reasons[b] = "act_halted"
                continue

            # 收敛
            if convergence_score[b] > (1 - self.convergence_threshold):
                # 检查是否多次循环无进步
                if loop_idx > self.max_loops_without_progress:
                    should_continue[b] = False
                    reasons[b] = "converged"
                    continue

            # 达到最大深度
            if loop_idx >= total_loops:
                should_continue[b] = False
                reasons[b] = "max_loops"
                continue

            # 深度建议早停
            if loop_idx >= recommended_depth:
                should_continue[b] = False
                reasons[b] = "depth_reached"
                continue

        return {
            "should_continue": should_continue,
            "recommended_depth": recommended_depth,
            "convergence_score": convergence_score,
            "act_cumulative_p": cumulative_p,
            "reason": reasons if B > 1 else reasons[0],
        }
