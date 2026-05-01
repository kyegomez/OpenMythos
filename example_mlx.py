import mlx.core as mx
from open_mythos.main import OpenMythos, MythosConfig

def test_run():
    # 1. テスト用の軽量設定
    # 動作確認のため、メモリ消費の少ない小さなモデルを定義します
    cfg = MythosConfig(
        vocab_size=1000,
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=128,
        max_loop_iters=4,
        attn_type="mla"  # MLA(Multi-Latent Attention)の動作確認
    )

    print("Initializing MLX OpenMythos model...")
    model = OpenMythos(cfg)

    # 2. ダミー入力の作成 (Batch=1, Seq=8)
    # MLXのランダムな整数配列を生成
    tokens = mx.random.randint(0, cfg.vocab_size, (1, 8))

    print(f"Input tokens shape: {tokens.shape}")

    # 3. フォワードパス（推論）の実行
    print("Running forward pass...")
    logits = model(tokens, n_loops=2)
    print(f"Logits shape: {logits.shape}")

    # 4. トークン生成のテスト
    print("Generating new tokens...")
    generated = model.generate(tokens, max_new_tokens=5, n_loops=2)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Sequence: {generated}")

    print("\n--- MLX Test Successful! ---")

if __name__ == "__main__":
    test_run()
