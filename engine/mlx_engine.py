import mlx.core as mx
from mlx_lm import load, generate

class MLXMythosEngine:
    def __init__(self, model_path: str):
        # Macのメモリ（Unified Memory）を効率的に使ってロード
        self.model, self.tokenizer = load(model_path)

    def generate_response(self, prompt: str, max_tokens: int = 512, temp: float = 0.7):
        # 推論の実行
        return generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            temp=temp
        )
