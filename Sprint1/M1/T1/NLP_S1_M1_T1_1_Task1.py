import numpy as np
import matplotlib.pyplot as plt
import torch

# Параметры
d_model = 30
seq_len = 100
batch_size = 2

# Синусно-косинусное позиционное кодирование
pos = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
i = torch.arange(0, d_model, 2)            # (d_model/2,)
angle_rates = 1 / torch.pow(10000, i / d_model)  # (d_model/2,)
pos_enc = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
pos_enc[:, 0::2] = torch.sin(pos * angle_rates)  # чётные индексы
pos_enc[:, 1::2] = torch.cos(pos * angle_rates)  # нечётные индексы

print(pos_enc)

# Визуализация
plt.figure(figsize=(12, 6))
plt.imshow(pos_enc, aspect='auto', cmap='viridis')
plt.xlabel('Размерность (d_model)')
plt.ylabel('Позиция в последовательности')
plt.title('Синусно-косинусное позиционное кодирование')
plt.colorbar(label='PE значение')
plt.tight_layout()
plt.show() 