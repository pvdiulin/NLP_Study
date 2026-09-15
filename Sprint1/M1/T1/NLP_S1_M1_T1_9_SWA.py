import torch, torch.nn.functional as F
import torch.nn as NN

def sliding_window_attention(Q, K, V, window_size):
    # Q,K,V формы (seq_len, batch, d_model)
    seq_len = Q.size(0)

    # Создаём маску формата (seq_len, seq_len) с -inf для запрещённых связей
    mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        left = max(0, i - window_size)
        right = min(seq_len, i + window_size + 1)
        mask[i, left:right] = 0

    # Переставляем размерности, чтобы батч был первым
    Q, K, V = Q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1) # shape: (batch, seq_len, d_model) 
    
    # Изменяем размер маски для соответствия батча размеру Q (shape: [batch, seq_len, seq_len])
    mask = mask.unsqueeze(0).expand(Q.size(0), -1, -1)

    # Используем scaled_dot_product_attention с маской
    attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

    # Возвращаем результат обратно в исходном порядке: (seq_len, batch, d_model)
    attn_out = attn_out.transpose(0, 1)

    return attn_out, mask

# Проверка маскированного внимания:
seq_len, batch, d_model = 10, 2, 16
Q = torch.randn(seq_len, batch, d_model)
out, mask = sliding_window_attention(Q, Q, Q, window_size=2)
print(out.shape)

# Дополнительная проверка визуализацией
print("\nПример маски для позиции 5 (первый батч):")
print(mask[0, 5].detach().numpy().round(1))