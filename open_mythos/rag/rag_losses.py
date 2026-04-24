"""
Phase 4: RAG 微调损失函数
==========================

检索质量信号回传的训练目标。

损失函数组成:
1. L_main: 主损失 (交叉熵)
2. L_retrieval: 检索质量损失
3. L_loop_efficiency: 循环效率损失
4. L_modality_routing: 模态路由损失
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Retrieval Quality Loss
# ============================================================================


class RetrievalQualityLoss(nn.Module):
    """
    检索质量损失。

    信号来源:
    1. 直接监督: 检索结果包含正确答案片段 → 正信号
    2. 间接监督: 最终答案正确 → REINFORCE 风格回传
    3. 对比学习: 正确检索 vs 随机负例
    """

    def __init__(
        self,
        direct_supervision_weight: float = 0.3,
        reinforce_weight: float = 0.5,
        contrastive_weight: float = 0.2,
        hit_threshold: float = 0.5,
    ):
        """
        Args:
            direct_supervision_weight: 直接监督权重
            reinforce_weight: REINFORCE 权重
            contrastive_weight: 对比学习权重
            hit_threshold: 命中阈值 (文本重叠度)
        """
        super().__init__()
        self.direct_weight = direct_supervision_weight
        self.reinforce_weight = reinforce_weight
        self.contrastive_weight = contrastive_weight
        self.hit_threshold = hit_threshold

    def forward(
        self,
        retrieved_entities: list[dict],
        answer: str,
        final_loss: torch.Tensor,
        loop_step: int,
        max_loops: int,
        reward_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        计算检索质量损失。

        Args:
            retrieved_entities: 每轮检索到的实体列表
            answer: 正确答案
            final_loss: 主损失
            loop_step: 当前循环步
            max_loops: 最大循环数
            reward_scale: 奖励缩放因子

        Returns:
            总损失 (包含检索质量信号)
        """
        total_loss = final_loss

        # 1. 直接监督: 检查检索结果是否包含答案
        hit = self._check_hit(retrieved_entities, answer)

        # 2. REINFORCE 风格的信用分配
        # 衰减因子: 越早的检索步骤权重越高
        decay = 0.9 ** (max_loops - loop_step)
        retrieval_reward = float(hit) * reward_scale

        # 负对数似然风格的奖励
        reinforce_loss = -decay * retrieval_reward

        # 3. 对比损失: 鼓励检索到正确答案的路径
        if retrieved_entities:
            contrastive_loss = self._contrastive_loss(retrieved_entities, answer)
        else:
            contrastive_loss = 0.0

        # 综合
        total_loss = total_loss + self.reinforce_weight * reinforce_loss
        if contrastive_loss != 0:
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        return total_loss

    def _check_hit(self, retrieved_entities: list[dict], answer: str) -> bool:
        """检查检索结果是否命中答案"""
        if not retrieved_entities:
            return False

        answer_lower = answer.lower()
        for entity in retrieved_entities:
            content = entity.get("content", "").lower()
            # 简单的文本重叠检查
            if self._text_overlap(content, answer_lower) > self.hit_threshold:
                return True

        return False

    def _text_overlap(self, text1: str, text2: str) -> float:
        """计算两个文本的重叠度"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _contrastive_loss(
        self,
        retrieved_entities: list[dict],
        answer: str,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        对比损失: 正确检索 vs 负例

        L = max(0, score_neg - score_pos + margin)
        """
        # 简化版本: 使用检索分数
        # 正例分数应该高，负例分数应该低
        scores = [e.get("score", 0.0) for e in retrieved_entities]

        if len(scores) < 2:
            return torch.tensor(0.0)

        # 最相关 vs 最不相关
        pos_score = max(scores)
        neg_score = min(scores)

        loss = max(0, neg_score - pos_score + margin)
        return torch.tensor(loss, requires_grad=True)


# ============================================================================
# Loop Efficiency Loss
# ============================================================================


class LoopEfficiencyLoss(nn.Module):
    """
    循环效率损失。

    鼓励模型:
    1. 早停 (如果已收敛)
    2. 避免无效循环
    3. 动态调整深度
    """

    def __init__(
        self,
        efficiency_weight: float = 0.05,
        convergence_threshold: float = 0.01,
    ):
        """
        Args:
            efficiency_weight: 效率损失权重
            convergence_threshold: 收敛阈值
        """
        super().__init__()
        self.efficiency_weight = efficiency_weight
        self.convergence_threshold = convergence_threshold

    def forward(
        self,
        loop_states: list[torch.Tensor],
        halting_probs: list[torch.Tensor],
        final_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算循环效率损失。

        Args:
            loop_states: 每轮的 hidden states
            halting_probs: 每轮的 halting 概率
            final_loss: 主损失

        Returns:
            总损失
        """
        total_loss = final_loss

        if len(loop_states) < 2:
            return total_loss

        # 1. 早停损失: 鼓励在收敛后停止
        early_stop_loss = self._early_stop_loss(loop_states, halting_probs)

        # 2. 深度正则化: 避免不必要的深层循环
        depth_reg_loss = self._depth_regularization(len(loop_states))

        total_loss = total_loss + self.efficiency_weight * (early_stop_loss + depth_reg_loss)

        return total_loss

    def _early_stop_loss(
        self,
        loop_states: list[torch.Tensor],
        halting_probs: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        早停损失: 当 hidden state 收敛时，应停止循环
        """
        loss = torch.tensor(0.0, device=loop_states[0].device)

        for t in range(len(loop_states) - 1):
            # 计算相邻两步的差异
            delta = torch.norm(loop_states[t + 1] - loop_states[t], dim=-1).mean()
            halting_p = halting_probs[t].mean()

            # 如果已收敛但还在继续循环，加惩罚
            if delta < self.convergence_threshold and halting_p < 0.9:
                loss = loss + delta  # 鼓励停止

        return loss / max(len(loop_states) - 1, 1)

    def _depth_regularization(self, actual_depth: int) -> torch.Tensor:
        """
        深度正则化: 惩罚过深的循环
        """
        # 鼓励使用较小的深度
        optimal_depth = 8
        penalty = abs(actual_depth - optimal_depth) / optimal_depth
        return torch.tensor(penalty, requires_grad=True)


# ============================================================================
# Modality Routing Loss
# ============================================================================


class ModalityRoutingLoss(nn.Module):
    """
    模态路由损失。

    鼓励模型:
    1. 正确路由到相关模态
    2. 使用适当的检索深度
    3. 模态决策与内容匹配
    """

    def __init__(
        self,
        routing_weight: float = 0.02,
    ):
        """
        Args:
            routing_weight: 路由损失权重
        """
        super().__init__()
        self.routing_weight = routing_weight

    def forward(
        self,
        routing_probs: torch.Tensor,  # (B, num_modalities)
        target_modality: Optional[str] = None,
        final_loss: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算模态路由损失。

        Args:
            routing_probs: 路由概率分布
            target_modality: 目标模态 (如果已知)
            final_loss: 主损失

        Returns:
            总损失
        """
        if final_loss is not None:
            total_loss = final_loss
        else:
            total_loss = torch.tensor(0.0, requires_grad=True, device=routing_probs.device)

        # 熵正则化: 鼓励路由决策不那么"随意"
        entropy = -(routing_probs * torch.log(routing_probs + 1e-8)).sum(dim=-1).mean()

        # 低熵 = 更确定的路由决策
        entropy_loss = -entropy  # 最大化熵 = 最小化这个损失

        total_loss = total_loss + self.routing_weight * entropy_loss

        return total_loss


# ============================================================================
# Combined RAG Loss
# ============================================================================


class CombinedRAGLoss(nn.Module):
    """
    组合 RAG 损失函数。

    综合:
    1. 主损失 (交叉熵)
    2. 检索质量损失
    3. 循环效率损失
    4. 模态路由损失
    """

    def __init__(
        self,
        retrieval_weight: float = 0.1,
        efficiency_weight: float = 0.05,
        routing_weight: float = 0.02,
    ):
        """
        Args:
            retrieval_weight: 检索损失权重
            efficiency_weight: 效率损失权重
            routing_weight: 路由损失权重
        """
        super().__init__()
        self.retrieval_loss = RetrievalQualityLoss()
        self.efficiency_loss = LoopEfficiencyLoss()
        self.routing_loss = ModalityRoutingLoss()

        self.retrieval_weight = retrieval_weight
        self.efficiency_weight = efficiency_weight
        self.routing_weight = routing_weight

    def forward(
        self,
        main_loss: torch.Tensor,
        retrieved_entities: list[dict],
        loop_states: list[torch.Tensor],
        halting_probs: list[torch.Tensor],
        routing_probs: torch.Tensor,
        answer: Optional[str] = None,
        loop_step: int = 0,
        max_loops: int = 16,
    ) -> torch.Tensor:
        """
        计算组合损失。

        Args:
            main_loss: 主损失 (交叉熵)
            retrieved_entities: 检索到的实体
            loop_states: 循环状态列表
            halting_probs:  halting 概率列表
            routing_probs: 路由概率
            answer: 正确答案 (可选)
            loop_step: 当前循环步
            max_loops: 最大循环数

        Returns:
            总损失
        """
        total_loss = main_loss

        # 检索质量损失
        if answer is not None and retrieved_entities:
            retrieval_loss = self.retrieval_loss(
                retrieved_entities, answer, main_loss, loop_step, max_loops
            )
            total_loss = total_loss + self.retrieval_weight * retrieval_loss

        # 循环效率损失
        if loop_states and halting_probs:
            efficiency_loss = self.efficiency_loss(
                loop_states, halting_probs, main_loss
            )
            total_loss = total_loss + self.efficiency_weight * efficiency_loss

        # 模态路由损失
        if routing_probs is not None:
            routing_loss = self.routing_loss(routing_probs, final_loss=main_loss)
            total_loss = total_loss + self.routing_weight * routing_loss

        return total_loss
