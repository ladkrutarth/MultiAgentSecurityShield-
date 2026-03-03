import os
from pathlib import Path
from mlx_lm import load, generate, sample_utils

class LocalLLM:
    """
    Wrapper for mlx-lm to provide a simple interface for local LLM generation.
    Highly optimized for Apple Silicon (M1/M2/M3 chips).
    """
    
    # Default high-quality Llama-3 8B Instruct model for MLX
    MODEL_ID = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id or self.MODEL_ID
        print(f"Loading local MLX model: {self.model_id}...")
        
        # MLX-LM handles downloading and loading automatically
        self.model, self.tokenizer = load(self.model_id)
        print("✅ Local MLX LLM loaded successfully.")

    def generate(self, prompt: str, max_tokens: int = 512, temp: float = 0.0) -> str:
        """Generate response from the local MLX model. Optimized for speed."""
        # Simple Instruct template for Llama-3
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # temp=0 is faster (greedy decoding) and more consistent for logic tasks
        sampler = sample_utils.make_sampler(temp=temp)
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        # Strip Llama-3 special tokens if they leak
        for token in ["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "assistant", "user"]:
            response = response.replace(token, "")

        response = response.strip()
        
        # Ensure response ends at the last complete sentence (period)
        if response and not response.endswith((".", "!", "?")):
            last_punct = max(response.rfind("."), response.rfind("!"), response.rfind("?"))
            if last_punct != -1:
                response = response[:last_punct + 1]

        return response

if __name__ == "__main__":
    # Quick test harness
    llm = LocalLLM()
    question = "What are the common indicators of credit card fraud?"
    print(f"\nQuestion: {question}")
    response = llm.generate(question)
    print(f"\nResponse: {response}")
