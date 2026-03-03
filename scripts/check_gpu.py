import mlx.core as mx
import sys

def check_gpu():
    print("System GPU Check...")
    print(f"Python Version: {sys.version}")
    print(f"MLX Version: {mx.__version__}")
    
    metal_available = mx.metal.is_available()
    print(f"Metal Available: {metal_available}")
    
    if metal_available:
        try:
            # Simple GPU operation test
            a = mx.array([1, 2, 3])
            b = mx.array([4, 5, 6])
            c = a + b
            mx.eval(c)
            print("✅ GPU Operation Test: Passed")
        except Exception as e:
            print(f"❌ GPU Operation Test: Failed ({e})")
    else:
        print("❌ Metal is NOT available. mlx-lm will fail.")

if __name__ == "__main__":
    check_gpu()
