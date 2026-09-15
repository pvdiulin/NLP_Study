import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = self.linear1(x)   # (seq_len, batch, dim_ff)
        x = self.relu(x)
        x = self.linear2(x)   # (seq_len, batch, d_model)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model) # <- Резидуальное соединение и нормализация
        self.ff = FeedForward(d_model, dim_ff)
        self.norm2 = nn.LayerNorm(d_model)  # <- Резидуальное соединение и нормализация

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        attn_out, _ = self.self_attn(x, x, x)     # Multi-Head Attention
        x = self.norm1(x + attn_out)              # Add & Norm
        ff_out = self.ff(x)                       # FeedForward
        x = self.norm2(x + ff_out)               # Add & Norm
        return x 

# d_model - размерность представления (embedding dimension)
# Это размер каждого вектора, который представляет один токен на протяжении всего трансформера.
# num_heads — количество голов внимания
# Механизм Multi-Head Attention делит каждый вектор размера d_model на несколько частей.
# dim_ff — размер скрытого слоя FeedForward

# Использование блока энкодера:
d_model, num_heads, dim_ff = 64, 8, 256
encoder_block = TransformerEncoderBlock(d_model, num_heads, dim_ff)
x = torch.randn(10, 2, d_model)          # (seq_len=10, batch_size=2, d_model=64)
y = encoder_block(x)
print(y.shape)  # (10, 2, 64) 