import torch
import math
import torch.nn as nn # модуль PyTorch, содержащий готовые слои нейронных сетей (Linear, Embedding, Transformer и др.).

d_model = 64 # Размерность эмбеддинга. Это означает, что каждый токен будет представлен вектором длиной 64.
seq_len = 5 # Длина последовательности. В предложении будет 5 токенов.
batch_size = 2 # Размер батча. Батч — это количество предложений, обрабатываемых одновременно.
tokens = torch.randint(0, 10000, (seq_len, batch_size))  # случайные токены
print(tokens.shape)

# -- Далее создайте слой эмбеддинга и прогоните через него свои токены, чтобы получить их векторные представления: --
emb_layer = nn.Embedding(num_embeddings=10000, embedding_dim=d_model) # num_embeddings - словарь содержит 10000 различных токенов (ID от 0 до 9999). / d_model - каждому токену соответствует вектор длиной 64.
x = emb_layer(tokens)  # форма (seq_len, batch_size, d_model)
print(x.shape) 

# --Синусно-косинусное позиционное кодирование--
pos = torch.arange(seq_len).unsqueeze(1)  # [[0],[1],...,[seq_len - 1]] 

# Для каждого четного k вычисляем, знаменатель угла,
# который потом пойдет под синус и косинус
k = torch.arange(0, d_model, 2) 
angle_rates = 1 / torch.pow(10000, k / d_model) # размер: (d_model / 2)

# Инициализируем матрицу позиционных кодов нулями
pos_enc = torch.zeros(seq_len, d_model)
# Cчитаем сами позиционные коды через sin и cos 
# для четных и нечетных индексов соответсвнно
pos_enc[:, 0::2] = torch.sin(pos * angle_rates)   # чётные индексы
pos_enc[:, 1::2] = torch.cos(pos * angle_rates)   # нечётные

# повторяем позиционные коды для каждой последовательности из батча
pos_enc = pos_enc.unsqueeze(1).repeat(1, batch_size, 1)  # (seq_len, batch_size, d_model) 
print(pos_enc.shape) 

h = x + pos_enc 
print(h.shape) 