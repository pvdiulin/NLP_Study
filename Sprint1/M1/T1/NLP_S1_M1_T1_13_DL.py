from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

def tokenize_batch(batch):
    texts  = [x['text'] for x in batch]
    labels = [x['label'] for x in batch]
    encoding = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(labels)
    }

def create_dataloader(dataset, batch_size=8, shuffle=True):
       return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=tokenize_batch
    )

imdb = load_dataset("imdb")
# Выберем 1000 примеров для обучения и 500 для теста
train_dataset = imdb['train'].shuffle(seed=42).select(range(1000))
test_dataset  = imdb['test'].shuffle(seed=42).select(range(500))

# Вывод первых 3 рецензий и меток
for i in range(3):
    print(f"Рецензия {i+1}: {train_dataset[i]['text']}")
    print(f"Метка {i+1}: {train_dataset[i]['label']}\n")

train_loader = create_dataloader(train_dataset)
test_loader  = create_dataloader(test_dataset, shuffle=False) 