import mlx.core as mx
from mlx_lm import load as mlx_load
from open_mythos.main import OpenMythos, MythosConfig, load_deepseek_v3_subset

def main():
    # 先ほど作成されたモデルパス
    mlx_path = "./models/deepseek-v2-mlx"
    
    print("--- Configuring OpenMythos for DeepSeek-V2/V3 ---")
    # DeepSeek-V2-Lite の実際のアーキテクチャに基づいた設定
    cfg = MythosConfig(
        vocab_size=102400,
        dim=2048,           # V2-Liteの隠れ層次元
        n_heads=16,
        attn_type="mla",    # Multi-Latent Attentionを有効化
        kv_lora_rank=512,
        n_experts=64,       # MoEのエキスパート数
        prelude_layers=2,   # 固定の Prelude 層
        max_loop_iters=4    # 再帰ループの回数（ここを増やすと深くなる）
    )
    
    model = OpenMythos(cfg)
    
    # 1. 実モデルの重みを OpenMythos の構造にロード
    print("Loading weights into OpenMythos structure...")
    load_deepseek_v3_subset(model, mlx_path)
    
    # 2. トークナイザーのロード
    print("Loading tokenizer...")
    _, tokenizer = mlx_load(mlx_path)
    
    # 3. 推論テスト
    prompt = "DeepSeek-V3 uses MLA because"
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    
    print(f"\nPrompt: {prompt}")
    print("Generating (Recurrent Loops: 4)...")
    
    # generateメソッドで推論実行
    output_ids = model.generate(input_ids, max_new_tokens=15, n_loops=4)
    
    # 結果のデコード
    response = tokenizer.decode(output_ids[0].tolist())
    print(f"\nResponse: {response}")
    print("\n--- DeepSeek Inference Successful! ---")

if __name__ == "__main__":
    main()
