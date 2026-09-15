import torch
from tqdm.auto import tqdm
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

epochs = 3
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        preds         = outputs.logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = total_correct / total_samples

    model.eval()
    val_loss, val_correct, val_samples = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch} [Eval]"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss    += outputs.loss.item()
            preds        = outputs.logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_samples += labels.size(0)

    val_loss = val_loss / len(test_loader)
    val_acc  = val_correct / val_samples

    print(f"\nEpoch {epoch} results:")
    print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    print(f"  Eval : loss={val_loss:.4f}, acc={val_acc:.4f}\n")

print(f"Final Eval Accuracy: {val_acc:.4f}") 