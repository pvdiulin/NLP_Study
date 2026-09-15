from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

num_epochs = 5
batch_size = 16  # Реальный batch size
learning_rate = 5e-5
accumulation_steps = 4  # Эффективный batch size = 16 * 4 = 64

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    n_batches = 0
    
    # Обнуляем градиенты в начале эпохи
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        # Переносим тензоры на device
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        # Масштабируем loss на количество шагов аккумуляции
        loss = loss / accumulation_steps
        
        loss.backward()
        
        # Обновляем параметры каждые accumulation_steps шагов
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Восстанавливаем реальное значение loss для статистики
        total_loss += loss.item() * accumulation_steps
        n_batches += 1
        
    # Обновляем параметры, если остались необработанные градиенты
    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    print(f"Epoch {epoch+1} effective optimizer steps: {(n_batches + accumulation_steps - 1) // accumulation_steps}")