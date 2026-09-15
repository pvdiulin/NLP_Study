import torch
import torch.nn as nn

# Реализация двухслойного FeedForward-блока
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

# Проверим, что FeedForward не меняет форму тензора
d_model, dim_ff = 64, 256
ff_block = FeedForward(d_model, dim_ff)
seq_len, batch_size = 5, 2
x = torch.randn(seq_len, batch_size, d_model)
y = ff_block(x)
print(x.shape, '->', y.shape)  # (5, 2, 64) -> (5, 2, 64) 

# В итоге - реализация блока FF, который сохраняет размерность входного тензора.
#  Теперь его можно использовать в энкодере трансформера.

