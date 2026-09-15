import torch
from transformers import AutoTokenizer, AutoModel

def get_model_size_mb(model):
    # Считаем размер всех параметров модели
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Считаем размер буферов модели (например, batch norm statistics)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Переводим из байт в мегабайты
    total_size_mb = (param_size + buffer_size) / 1024 / 1024
    return total_size_mb

# Тестируем функцию на разных моделях
models_to_test = [
    "distilbert-base-uncased",
    "cointegrated/rubert-tiny",
    "microsoft/MiniLM-L12-H384-uncased"
]

print("Сравнение размеров моделей")
print("=" * 40)

for model_name in models_to_test:
    print(f"\nЗагружаем {model_name}...")
    
    # Загружаем модель
    model = AutoModel.from_pretrained(model_name)
    
    # Вычисляем размер
    size_mb = get_model_size_mb(model)
    
    # Считаем количество параметров
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"Размер: {size_mb:.1f} МБ")
    print(f"Параметры: {num_params:.1f}")
    print("-" * 40)