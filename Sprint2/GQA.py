import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, dropout=0.0, bias=True):
        """
        embed_dim: общий размер эмбеддинга (D)
        num_heads: число query-гoлoв (H)
        num_groups: число групп для ключей/значений (G)
        """
        super().__init__()
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Проекции: Q — на все головы, K/V — только для групп
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_groups * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_groups * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.head_dim)
        # сколько query-гoлoв на группу
        self.heads_per_group = num_heads // num_groups

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x: (B, L, D)
        attn_mask: (B, 1, L, L)
        """
        B, L, _ = x.size()

        # Q: (B, L, H * head_dim) -> (B, H, L, head_dim)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)

        # K/V: (B, L, G * head_dim) -> (B, G, L, head_dim)
        k = self.k_proj(x).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)  # (B, G, L, d)
        v = self.v_proj(x).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)  # (B, G, L, d)

        # нельзя перемножить (B, H, L, d) на (B, G, d, L) 
        # k, v: (B, G, L, d) -> повторим по heads_per_group, чтобы соответствовало q
        # сначала сделаем (B*G, L, d)
        k = k.reshape(B * self.num_groups, L, self.head_dim)
        v = v.reshape(B * self.num_groups, L, self.head_dim)

        # расширяем по количеству голов в группе: (B*G, 1, L, d) -> (B*G, heads_per_group, L, d)
        k_expanded = k.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1)
        v_expanded = v.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1)
        # и приводим к (B, H, L, d)
        k_expanded = k_expanded.reshape(B, self.num_groups * self.heads_per_group, L, self.head_dim)
        v_expanded = v_expanded.reshape(B, self.num_groups * self.heads_per_group, L, self.head_dim)
        # Attention: Q @ K^T
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3))  # (B, H, L, L)
        attn_weights = attn_weights * self.scale

        # Применяем маску
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_probs = F.softmax(attn_weights, dim=-1)  # (B, H, L, L)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v_expanded)  # (B, H, L, d)

        # Собираем обратно: (B, G, heads_per_group, L, d) -> (B, H, L, d)
        attn_output = attn_output.view(B, self.num_groups, self.heads_per_group, L, self.head_dim)
        attn_output = attn_output.reshape(B, self.num_heads, L, self.head_dim)

        # объединяем головы
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)  # (B, L, D)

        out = self.out_proj(attn_output)  # (B, L, D)
        return out

# Пример: входной тензор случайный
batch_size = 2
seq_len = 10
embed_dim = 64
num_heads = 16
num_groups = 4  # по 4 головы делят 1 набор K/V

model = GroupedQueryAttention(embed_dim=embed_dim, num_heads=num_heads, num_groups=num_groups)
x = torch.randn(batch_size, seq_len, embed_dim)

causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))  # (L, L)
causal_mask = causal_mask.unsqueeze(0)  

print("Input shape:", x.shape)
out = model(x, attn_mask=causal_mask)  # (B, L, D)
print("Output shape:", out.shape)