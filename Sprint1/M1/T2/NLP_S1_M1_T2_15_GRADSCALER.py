from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
# Импорты для AMP
from torch.amp import GradScaler, autocast

num_epochs = 5
batch_size = 16
learning_rate = 5e-5

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Инициализируем GradScaler для масштабирования градиентов
scaler = GradScaler()

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        # Переносим тензоры на device
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Используем autocast для автоматического приведения к float16
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
        
        # Масштабируем loss и выполняем backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")