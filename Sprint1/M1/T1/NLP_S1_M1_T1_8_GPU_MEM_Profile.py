from transformers import BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()

# Проверка работы GPU
assert torch.cuda.is_available(), "Ошибка: GPU недоступна!"

for seq_len in [128,256,512]:
    x = torch.randint(0, model.config.vocab_size, (1, seq_len)).cuda()
    torch.cuda.reset_peak_memory_stats()
    _ = model(x)
    print(f"Длина {seq_len}: пиковое потребление ≈ {torch.cuda.max_memory_allocated()/1024**2:.0f} МБ")