import time
from transformers import pipeline

def measure_inference_time(classifier, texts, num_runs=10):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        # Прогоняем все тексты через классификатор
        for text in texts:
            _ = classifier(text)
        end_time = time.time()
        # Сохраняем время выполнения
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    # Среднее время на один текст
    avg_time_per_text = avg_time / len(texts)
    return avg_time_per_text * 1000  # в миллисекундах

# Тестовые тексты на английском
test_texts = [
    "This product is amazing!", # Этот продукт потрясающий!
    "Terrible quality, very disappointed.", # Ужасное качество, очень разочарован.
    "The service was okay, nothing special.", # Сервис был нормальным, ничего особенного.
    "Outstanding experience! Highly recommend!", # Выдающийся опыт! Настоятельно рекомендую!
    "Poor customer support, took forever." # Плохая поддержка клиентов, всё заняло вечность.
]

# Список англоязычных моделей для тестирования
models_to_test = [
    "distilbert-base-uncased",
    "microsoft/MiniLM-L12-H384-uncased"
]

for model_name in models_to_test:
    print(f"\nТестирование скорости: {model_name}")
    print("=" * 50)
    
    # Создаём pipeline для модели
    classifier = pipeline("text-classification", model=model_name)
    
    # Измеряем среднее время инференса
    avg_time = measure_inference_time(classifier, test_texts)
    
    print(f"Среднее время обработки одного текста: {avg_time:.3f} мс")