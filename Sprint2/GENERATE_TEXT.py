import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

def generate(model, tokenizer, prompt, max_new_tokens=10):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    generated = input_ids # сначала сгенерированный текст = начальному

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated) # прогоняем входные токены через модель
            next_token_logits = outputs.logits[:, -1, :] # извлекаем логиты только последнего токена
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # выбираем следующий токен как токен с наибольшей вероятностью
            generated = torch.cat((generated, next_token), dim=1) # добавляем новый токен к последовательности

    return tokenizer.decode(generated[0])  # возвращаем декодированный текст

print(generate(model, tokenizer, 'Привет,'))