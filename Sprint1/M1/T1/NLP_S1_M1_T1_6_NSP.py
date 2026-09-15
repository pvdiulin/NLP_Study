# В задаче NSP модель получает две текстовые последовательности (A и B) и прогнозирует, 
# идёт ли B непосредственно после A в оригинальном тексте. Обучающая выборка формируется так,
#  что половина пар A–B являются соседними предложениями из корпуса (метка IsNext), 
# а половина — случайными (метка NotNext). На вход в модель поступает [CLS] ... [SEP] ... [SEP], 
# а выходное представление тега [CLS] поступает в двоичный классификатор. 

import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

# Инициализация модели и токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.to("cuda")

# Пример для IsNext (последовательные предложения)
text_a = "The cat sat on the mat"
text_b = "It was very sleepy"

# Токенизация и подготовка входа
inputs = tokenizer(text_a, text_b, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()} # Перенос всех входных тензоров на GPU так как модель находится на GPU

# Получение предсказаний
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Интерпретация результатов
is_next_prob = torch.softmax(logits, dim=1)[0][1].item()  # Вероятность NotNext
not_next_prob = torch.softmax(logits, dim=1)[0][0].item() # Вероятность IsNext

print(f"Вероятность IsNext: {not_next_prob:.4f}")
print(f"Вероятность NotNext: {is_next_prob:.4f}")
print("\n" + "="*80 + "\n") 