import os
import anyio
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

    def generate(self, prompt: str, max_tokens: int = 250, temp: float = 0.0) -> str:
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

    async def generate_async(self, prompt: str, max_tokens: int = 250, temp: float = 0.0) -> str:
        """Run generation in a separate thread to avoid blocking the event loop."""
        return await anyio.to_thread.run_sync(self.generate, prompt, max_tokens, temp)


class FastDeceptionLLM:
    """
    Lightweight, fast LLM for deception grid only. Uses a smaller model so
    security/deception path stays quick and does not block the main GuardAgent.
    Set DECEPTION_LLM_MODEL env to override; set DECEPTION_LLM_DISABLED=1 to use templates only.
    """
    # Small 4-bit model for speed (alternative to main Llama-3 8B)
    DEFAULT_MODEL_ID = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"

    def __init__(self, model_id: str = None):
        self.model_id = (os.environ.get("DECEPTION_LLM_MODEL") or model_id or self.DEFAULT_MODEL_ID).strip()
        self._model = None
        self._tokenizer = None
        if os.environ.get("DECEPTION_LLM_DISABLED", "").strip() == "1":
            self.model_id = None  # templates only
            return
        try:
            self._model, self._tokenizer = load(self.model_id)
            print("✅ Fast Deception LLM loaded (security path).")
        except Exception as e:
            print(f"⚠️ Fast Deception LLM not loaded ({e}); using templates only.")
            self.model_id = None

    def generate(self, prompt: str, max_tokens: int = 150, temp: float = 0.7) -> str:
        """Short, fast generation for decoy responses."""
        if self._model is None or self._tokenizer is None:
            return ""
        formatted = f"<|system|>\nYou reply briefly and professionally.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        sampler = sample_utils.make_sampler(temp=temp)
        out = generate(
            self._model, self._tokenizer, prompt=formatted, max_tokens=max_tokens,
            sampler=sampler, verbose=False
        )
        for t in ["</s>", "<|assistant|>", "<|user|>", "<|system|>"]:
            out = out.replace(t, "")
        return out.strip() or "Request received. Our team will follow up shortly."

    async def generate_async(self, prompt: str, max_tokens: int = 150, temp: float = 0.7) -> str:
        return await anyio.to_thread.run_sync(self.generate, prompt, max_tokens, temp)


if __name__ == "__main__":
    # Quick test harness (runs synchronously)
    llm = LocalLLM()
    question = "What are the common indicators of credit card fraud?"
    print(f"\nQuestion: {question}")
    response = llm.generate(question)
    print(f"\nResponse: {response}")
