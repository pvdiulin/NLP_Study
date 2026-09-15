from datasets import load_dataset
from torch.utils.data import DataLoader

imdb = load_dataset("imdb")
# Выберем 1000 примеров для обучения и 500 для теста
train_dataset = imdb['train'].shuffle(seed=42).select(range(1000))
test_dataset  = imdb['test'].shuffle(seed=42).select(range(500))

# Вывод первых 3 рецензий и меток
for i in range(3):
    print(f"Рецензия {i+1}: {train_dataset[i]['text']}")
    print(f"Метка {i+1}: {train_dataset[i]['label']}\n")