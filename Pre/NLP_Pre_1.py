from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")

# Перевод модели в режим оценки
model.eval()

print("Модель и токенизатор успешно загружены!")

# Пример текста на русском языке
text = "Я остался недоволен этим товаром"

# Токенизация с использованием padding и truncation
inputs = tokenizer(
    text,
    padding='max_length',   # Все последовательности дополняются специальными токенами до заданной длины. Это необходимо, чтобы обеспечить одинаковый размер входа для всех примеров.
    truncation=True,        # Обрезка, если текст длиннее max_length
    max_length=64,          # Если текст превышает 64 токена, он обрезается до нужного размера, чтобы избежать ошибок при обработке.
    return_tensors="pt"     # Результат возвращается в формате тензоров PyTorch, что удобно для дальнейшей работы с моделью.
)

# Выполнение предсказания без вычисления градиентов
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Применяем softmax для получения вероятностей
probabilities = torch.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities).item()

# Выводим результат
print(f"\nПредсказанный класс: {predicted_class}")
print(f"Вероятности классов: {probabilities}")