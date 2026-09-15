from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Загружаем токенизатор и модель XLM-R для классификации тональности
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Создаём pipeline для классификации
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Тестовые тексты на разных языках
texts = [
    "This movie is absolutely fantastic! I loved every moment of it.",  # английский
    "Этот фильм просто ужасен, потратил время зря.",  # русский  
    "¡Me encanta este producto! Es increíble y muy útil.",  # испанский
    "I'm not sure about this book, it's okay I guess.",  # английский
    "Сервис отличный, всем рекомендую!",  # русский
    "No me gusta nada, muy decepcionante."  # испанский
]

# Классифицируем тональность для каждого текста
for i, text in enumerate(texts):
    result = classifier(text)
    print(f"Текст {i+1}: {text}")
    print(f"Тональность: {result[0]['label']} (уверенность: {result[0]['score']:.3f})")
    print("-" * 50) 