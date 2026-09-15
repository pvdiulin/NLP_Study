from transformers import BertTokenizer, BertForMaskedLM
import torch

# Ключевая обучающая задача — Masked Language Modeling (MLM), то есть моделирование маскированного языка. Это как игра «Угадай слово по контексту слева и справа». «Играют» в неё так:
# 1. Сначала случайным образом маскируется 15% слов в предложении. 15% — это оптимальный баланс: достаточно, чтобы модель училась, но не слишком много, чтобы контекст для предсказания не разрушался.
# Маскирование не всегда происходит одинаково: 80% заменяется на [MASK], 10% — на случайный токен, 10% — остаются без изменений. 
# 2. Затем модель предсказывает исходные слова на местах масок, анализируя весь контекст вокруг них — и слева, и справа. 
# Такой подход учит BERT понимать двунаправленные связи между словами.

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to("cuda")

sentence = "Paris is the capital of [MASK]."
inputs = tokenizer(sentence, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()} # Перенос всех входных тензоров на GPU так как модель находится на GPU

# Используйте позицию [MASK] и получите предсказание
mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

# Выполните предсказание и получите логиты
with torch.no_grad(): 
    outputs = model(**inputs)
    logits = outputs.logits

# Выберите токен с наивысшим логитом в позиции маски с помощью argmax и item
predicted_index = logits[0, mask_index].argmax().item()
print(tokenizer.decode(predicted_index))

# Этот пример демонстрирует механизм MLM: даже без дообучения модель BERT уже умеет угадывать замаскированные слова по контексту.