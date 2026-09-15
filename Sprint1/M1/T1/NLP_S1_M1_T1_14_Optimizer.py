import torch
from datasets import load_dataset
from transformers import BertConfig, BertForSequenceClassification
from torch.optim import AdamW


# Устройство: GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Конфигурация и модель
config = BertConfig.from_pretrained('google-bert/bert-base-uncased', num_labels=2)
model  = BertForSequenceClassification.from_pretrained(
    'google-bert/bert-base-uncased', config=config
).to(device)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=2e-5) 