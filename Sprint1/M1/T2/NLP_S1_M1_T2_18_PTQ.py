import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = 'cpu'
model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
model_cpu = model.to(device)

text = 'This is a sample sentence for quantization benchmark.'
inputs = tokenizer(text, return_tensors='pt')

n_runs = 50

with torch.inference_mode():
    # исходный инференс
    start = time.time()
    for _ in range(n_runs):
        _ = model_cpu(**inputs)
    t_orig = (time.time() - start) / n_runs
print(f'FP32 avg inference time (per run): {t_orig:.6f} s')

print('Применяем динамическую квантизацию...')
# Нужно применить quantize_dynamic к model_cpu (квантуем torch.nn.Linear)
quantized_model = torch.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)

quantized_model.eval()

with torch.inference_mode():
    start = time.time()
    for _ in range(n_runs):
        _ = quantized_model(**inputs)
    t_q = (time.time() - start) / n_runs
print(f'Quantized avg inference time (per run): {t_q:.6f} s')

with torch.inference_mode():
    logits_fp32 = model_cpu(**inputs).logits.detach()
    logits_q = quantized_model(**inputs).logits.detach()

print('Примеры логитов (FP32 vs Quantized):')
print(logits_fp32[0][:6].tolist())
print(logits_q[0][:6].tolist())

# Посчитать L2 и max-abs разницу между logits_fp32 и logits_q.
# Формула L2: l2 = ||logits_fp32 - logits_q||_2
# max_abs = max(abs(logits_fp32 - logits_q))
diff = (logits_fp32 - logits_q).cpu()
l2 = torch.norm(diff).item()
max_abs = torch.max(torch.abs(diff)).item()
print(f'L2 diff: {l2:.6f}, max abs diff: {max_abs:.6f}')

# Сохранение state_dict'ов и сравнение размеров
tmp_fp = 'model_fp32.pth'
tmp_q = 'model_q.pth'
torch.save(model_cpu.state_dict(), tmp_fp)
torch.save(quantized_model.state_dict(), tmp_q)
print('FP32 size (MB):', os.path.getsize(tmp_fp)/1024/1024)
print('Quant size (MB):', os.path.getsize(tmp_q)/1024/1024)