import torch.nn as nn
import torch

class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 1) d_model должно делиться на num_heads:
        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"
        
        # 2) Сохраняем число голов и размерность каждой головы:
        self.num_heads = num_heads
        self.head_dim   = d_model // num_heads

        # 3) Линейные слои для проекции входных эмбеддингов
        #    в Q, K, V:
        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)

        # 4) Линейный слой для объединённого выхода всех голов:
        self.WO = nn.Linear(d_model, d_model, bias=False) 

    def forward(self, query, key, value, mask=None):
        # query, key, value: (seq_len, batch, d_model)
        seq_len, batch, d_model = query.shape

        # Проекция Q, K, V
        Q = self.WQ(query)  # (seq_len, batch, d_model)
        K = self.WK(key)
        V = self.WV(value)

        # Переставим размерности так, чтобы голова была отдельным измерением
        # Сначала делаем (batch, seq_len, d_model)
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)
        # Затем разбиваем по головам и снова транспонируем:
        # (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Скалярное произведение и softmax
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, seq, seq)
        # Если нужно запретить некоторым парам токенов «видеть» друг друга 
        # (например, будущие позиции при генерации), накладываем маску:
        # mask имеет форму (seq_len, seq_len), где 0 = «скрыть связь».
        # Расширяем до (1, 1, seq_len, seq_len) и подставляем −∞ в позиции с нулём.
        if mask is not None:
            # Расширяем маску до (batch, heads, seq, seq) и обнуляем запрещённые связи
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1,1,seq,seq)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)  # (batch, heads, seq, seq)

        # Взвешенное суммирование значений
        out = torch.matmul(attention, V)  # (batch, heads, seq, head_dim)

        # Объединяем головы обратно: (batch, seq_len, d_model)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, d_model)
        # Финальный линейный слой
        out = self.WO(out)  # (batch, seq_len, d_model)
        # Возвращаем в формат (seq_len, batch, d_model)
        out = out.permute(1, 0, 2)
        return out 

# Использование класса MyMultiHeadAttention:
d_model, num_heads = 64, 8
mha = MyMultiHeadAttention(d_model, num_heads)
x = torch.randn(10, 2, d_model)    # (seq_len=10, batch=2, d_model=64)
y = mha(x, x, x, mask=None) 
print(y.shape)  # (10, 2, 64) 