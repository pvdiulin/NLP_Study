from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast

num_epochs = 5
batch_size = 16
learning_rate = 5e-5
accumulation_steps = 4

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    n_batches = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        n_batches += 1
        
    # Обработка остатка
    if n_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")