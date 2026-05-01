import mlx.core as mx
import mlx.nn as nn
from open_mythos.main import OpenMythos, MythosConfig
# 既存のOpenMythosクラスがtorch.nn.Moduleを継承している場合、
# 本来はMLX用にモデル定義自体を書き換える必要がありますが、
# ここでは「MLXの演算体系」に合わせた検証用コードとして提示します。

attn_type = "mla"  # or "gqa"

base = {
    "vocab_size": 1000,
    "dim": 256,
    "n_heads": 8,
    "max_seq_len": 128,
    "max_loop_iters": 4,
    "prelude_layers": 1,
    "coda_layers": 1,
    "n_experts": 8,
    "n_shared_experts": 1,
    "n_experts_per_tok": 2,
    "expert_dim": 64,
    "lora_rank": 8,
    "attn_type": attn_type,
}

if attn_type == "gqa":
    cfg = MythosConfig(**base, n_kv_heads=2)
else:
    cfg = MythosConfig(
        **base,
        n_kv_heads=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )

# 1. モデルの初期化
# 注: OpenMythosがtorchベースの場合、本来はmlx.nn.Moduleへの移植が必要です。
# ここでは構造の互換性を確認します。
model = OpenMythos(cfg)

# 2. パラメータ数のカウント (MLX流)
# MLXではパラメータは辞書形式やツリー形式で管理されるため、
# torchのparameters()とは取得方法が異なりますが、今回は構造確認を優先します。
print(f"\n[{attn_type.upper()}] Initialized for MLX test environment")

# 3. ダミーデータの生成 (torch.randint -> mx.random.randint)
# MLXは(Batch, Seq)の形式をそのまま扱えます。
ids = mx.random.randint(0, cfg.vocab_size, (2, 16))

# 4. 推論実行
# model自体がMLX対応(mlx.nn.Module継承)している必要があります。
# 未対応の場合は以下の実行でエラーが出るため、その場合はモデル定義の移植へ進みます。
try:
    logits = model(ids, n_loops=4)
    print(f"[{attn_type.upper()}] Logits shape: {logits.shape}")

    # 5. 生成テスト
    out = model.generate(ids, max_new_tokens=8, n_loops=8)
    print(f"[{attn_type.upper()}] Generated shape: {out.shape}")

    # 6. スペクトル半径の確認 (A.max().item() -> mx.max(A).item())
    A = model.recurrent.injection.get_A()
    max_radius = mx.max(A).item()
    print(
        f"[{attn_type.upper()}] Spectral radius ρ(A) max: {max_radius:.4f} (must be < 1)"
    )

except TypeError as e:
    print(f"\n[ERROR] OpenMythos class is still based on PyTorch.")
    print("To run on MLX, we need to port 'open_mythos/main.py' to use 'mlx.nn.Module'.")
    print(f"Original Error: {e}")

