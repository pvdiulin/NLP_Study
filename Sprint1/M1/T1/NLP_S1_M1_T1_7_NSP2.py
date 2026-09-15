import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.to("cuda")

# Пример для NotNext (не последовательные предложения)
text_a =  "The cat sat on the mat"
text_b = "The Eiffel Tower is located in Paris"

# Токенизация и подготовка входа
inputs = tokenizer(text_a, text_b, return_tensors='pt')
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