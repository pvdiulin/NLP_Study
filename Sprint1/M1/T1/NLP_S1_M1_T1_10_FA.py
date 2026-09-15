import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())

# Параметры теста
batch, heads, seq_len, head_dim = 8, 4, 512, 32
dtype = torch.float16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Случайные Q,K,V
query = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
key   = torch.randn_like(query)
value = torch.randn_like(query)

# Функция для бенчмарка (в мкс)
import torch.utils.benchmark as benchmark
def bench(f, *args):
    t0 = benchmark.Timer(stmt="f(*args)", globals={"f": f, "args": args})
    return t0.blocked_autorange().mean * 1e6

print("Тестовый запуск...")

# Обычная реализация (Math)
with sdpa_kernel(SDPBackend.MATH):
    t_math = bench(F.scaled_dot_product_attention, query, key, value)
print(f"Math implementation: {t_math:.1f} мкс")

# Flash Attention реализация
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        t_flash = bench(F.scaled_dot_product_attention, query, key, value)
        print(f"Flash Attention impl.: {t_flash:.1f} мкс")
    except RuntimeError as e:
        print("Flash Attention не поддерживается:", e)

# Память-эффективная реализация
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    t_eff = bench(F.scaled_dot_product_attention, query, key, value)
print(f"Memory-Efficient impl.: {t_eff:.1f} мкс") 