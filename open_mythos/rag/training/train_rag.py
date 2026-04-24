"""
Phase 4: 两阶段训练脚本
=========================

Stage 1: 基础预训练
    - 标准 LM 目标
    - 基础循环推理
    - 模态嵌入初始化

Stage 2: RAG 微调
    - 冻结主干，训练检索相关模块
    - 检索质量损失
    - 多模态路由微调
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from open_mythos.rag.training import CombinedRAGLoss


# ============================================================================
# Config
# ============================================================================


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    train_data_path: str = "./data/train.jsonl"
    val_data_path: str = "./data/val.jsonl"
    max_seq_len: int = 4096
    max_loop_iters: int = 16

    # 训练
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Stage 1
    stage1_steps: int = 50000
    stage1_lr: float = 5e-5

    # Stage 2 (RAG)
    stage2_steps: int = 50000
    stage2_lr: float = 1e-4

    # 检索
    retrieval_top_k: int = 10
    retrieval_depth: int = 3

    # 损失权重
    retrieval_loss_weight: float = 0.1
    efficiency_loss_weight: float = 0.05
    routing_loss_weight: float = 0.02

    # 模型
    model_path: str = "./checkpoints/mythos_base.pt"
    output_dir: str = "./checkpoints/rag"

    # 硬件
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


# ============================================================================
# Dataset
# ============================================================================


class RAGDataset(torch.utils.data.Dataset):
    """
    RAG 训练数据集。

    每条数据格式:
    {
        "id": "doc_001",
        "question": "What is the main finding?",
        "answer": "The main finding is...",
        "contexts": [
            {"type": "text", "content": "...", "page_idx": 0},
            {"type": "image", "img_path": "...", "caption": "..."},
            ...
        ],
        "modalities": ["text", "image"],  # 主要模态
        "complexity": 0.5,  # 0-1 复杂度
    }
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 4096,
        tokenizer: Optional[Any] = None,
    ):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        # 加载数据
        self.samples = self._load_data()

    def _load_data(self) -> list[dict]:
        """加载数据"""
        import json

        samples = []
        with open(self.data_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        return {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "contexts": sample.get("contexts", []),
            "modalities": sample.get("modalities", ["text"]),
            "complexity": sample.get("complexity", 0.5),
        }


# ============================================================================
# Training
# ============================================================================


class TwoStageRAGTrainer:
    """
    两阶段 RAG 训练器。

    Stage 1: 基础预训练 (可选，从已有 checkpoint 继续)
    Stage 2: RAG 微调
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[RAGDataset] = None,
        val_dataset: Optional[RAGDataset] = None,
    ):
        """
        Args:
            model: OpenMythosRAG 模型
            config: 训练配置
            train_dataset: 训练数据集
            val_dataset: 验证数据集
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.device = config.device
        self.global_step = 0

        # 损失函数
        self.rag_loss = CombinedRAGLoss(
            retrieval_weight=config.retrieval_loss_weight,
            efficiency_weight=config.efficiency_loss_weight,
            routing_weight=config.routing_loss_weight,
        )

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self):
        """执行完整两阶段训练"""
        print("=" * 60)
        print("Stage 1: 基础预训练 (或从 checkpoint 加载)")
        print("=" * 60)

        # Stage 1: 基础预训练 (如果需要)
        if self.global_step < self.config.stage1_steps:
            self._stage1_pretrain()

        print("\n" + "=" * 60)
        print("Stage 2: RAG 微调")
        print("=" * 60)

        # Stage 2: RAG 微调
        self._stage2_rag_finetune()

        print("\n训练完成!")

    def _stage1_pretrain(self):
        """Stage 1: 基础预训练"""
        # 简化: Stage 1 主要训练主干模型
        # 这里假设已经有一个预训练好的模型

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.stage1_lr,
            weight_decay=self.config.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.stage1_steps,
            eta_min=self.config.min_lr,
        )

        self._training_loop(
            optimizer=optimizer,
            scheduler=scheduler,
            max_steps=self.config.stage1_steps,
            stage_name="Stage 1",
        )

    def _stage2_rag_finetune(self):
        """Stage 2: RAG 微调"""
        # 冻结主干，训练检索相关模块
        self._freeze_backbone()

        # 需要训练的参数
        trainable_params = list(self.model.rag_block.parameters())
        trainable_params += list(self.model.joint_selector.parameters())

        optimizer = AdamW(
            trainable_params,
            lr=self.config.stage2_lr,
            weight_decay=self.config.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.stage2_steps,
            eta_min=self.config.min_lr,
        )

        self._training_loop(
            optimizer=optimizer,
            scheduler=scheduler,
            max_steps=self.config.stage2_steps,
            stage_name="Stage 2 (RAG)",
        )

    def _freeze_backbone(self):
        """冻结主干模型"""
        # 冻结 transformer_blocks
        for block in self.model.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = False

        # 冻结 embedding
        if hasattr(self.model, "token_embedding"):
            for param in self.model.token_embedding.parameters():
                param.requires_grad = False

    def _training_loop(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        max_steps: int,
        stage_name: str,
    ):
        """通用训练循环"""
        self.model.train()

        train_loader = self._get_dataloader(self.train_dataset)
        val_loader = self._get_dataloader(self.val_dataset) if self.val_dataset else None

        epoch = 0
        while self.global_step < max_steps:
            epoch += 1
            print(f"\n{stage_name} - Epoch {epoch}")

            for batch in train_loader:
                # 检查是否达到最大步数
                if self.global_step >= max_steps:
                    break

                # 前向传播
                loss, info = self._training_step(batch)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                self.global_step += 1

                # 日志
                if self.global_step % 100 == 0:
                    print(f"  Step {self.global_step}/{max_steps} | Loss: {loss.item():.4f}")
                    self._log_info(info)

                # 验证
                if val_loader and self.global_step % 1000 == 0:
                    val_loss = self._validate(val_loader)
                    print(f"  Validation Loss: {val_loss:.4f}")

                # 保存 checkpoint
                if self.global_step % 5000 == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

        # 最终保存
        self._save_checkpoint("final")

    def _training_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """单步训练"""
        question = batch["question"]
        answer = batch["answer"]
        contexts = batch.get("contexts", [])
        modalities = batch.get("modalities", ["text"])

        # 获取设备上的数据
        question = question.to(self.device)
        answer = answer.to(self.device)

        # 前向传播 (通过模型)
        # 简化: 假设模型接受 text 输入
        # 生产版本需要处理多模态
        output = self.model(
            input_ids=question,
            labels=answer,
            retrieval_contexts=contexts,
            target_modalities=modalities,
        )

        loss = output["loss"]
        info = {
            "retrieved_entities": output.get("retrieved_entities", []),
            "loop_states": output.get("loop_states", []),
            "halting_probs": output.get("halting_probs", []),
            "routing_probs": output.get("routing_probs", None),
        }

        # RAG 损失
        if info["retrieved_entities"] and info["loop_states"]:
            loss = self.rag_loss(
                main_loss=loss,
                retrieved_entities=info["retrieved_entities"],
                loop_states=info["loop_states"],
                halting_probs=info["halting_probs"],
                routing_probs=info["routing_probs"],
                answer=batch.get("answer_text", ""),
                loop_step=len(info["loop_states"]),
                max_loops=self.config.max_loop_iters,
            )

        return loss, info

    def _validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                question = batch["question"].to(self.device)
                answer = batch["answer"].to(self.device)

                output = self.model(
                    input_ids=question,
                    labels=answer,
                )

                total_loss += output["loss"].item()
                num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _get_dataloader(self, dataset: Optional[RAGDataset]) -> Optional[DataLoader]:
        """创建 DataLoader"""
        if dataset is None:
            return None

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            pin_memory=self.config.pin_memory,
        )

    def _log_info(self, info: dict):
        """记录训练信息"""
        if info.get("retrieved_entities"):
            num_retrieved = len(info["retrieved_entities"])
            print(f"    Retrieved: {num_retrieved} entities")

        if info.get("halting_probs"):
            avg_p = sum(info["halting_probs"]) / len(info["halting_probs"])
            print(f"    Avg halting prob: {avg_p:.3f}")

    def _save_checkpoint(self, name: str):
        """保存 checkpoint"""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"rag_{name}.pt"
        )

        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")


# ============================================================================
# 快速开始脚本
# ============================================================================


def main():
    """快速开始训练"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Training")
    parser.add_argument("--model_path", type=str, default="./checkpoints/mythos_base.pt")
    parser.add_argument("--train_data", type=str, default="./data/train.jsonl")
    parser.add_argument("--val_data", type=str, default="./data/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/rag")
    parser.add_argument("--stage1_steps", type=int, default=0)  # 跳过 Stage 1
    parser.add_argument("--stage2_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    # 配置
    config = TrainingConfig(
        model_path=args.model_path,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        batch_size=args.batch_size,
        stage2_lr=args.lr,
    )

    # 加载模型
    print("加载模型...")
    from open_mythos import OpenMythosRAG

    model = OpenMythosRAG.load(args.model_path)
    model = model.to(config.device)

    # 数据集
    train_dataset = RAGDataset(config.train_data_path) if os.path.exists(config.train_data_path) else None
    val_dataset = RAGDataset(config.val_data_path) if os.path.exists(config.val_data_path) else None

    # 训练器
    trainer = TwoStageRAGTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
