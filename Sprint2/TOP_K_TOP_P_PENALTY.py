import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Для воспроизводимости результатов можно зафиксировать сид
torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model.eval()


@torch.no_grad()
def sample_next_token(logits, generated_ids, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0):
    # 1) Применяем штраф повторов: уменьшаем логиты для уже встречавшихся токенов
    if repetition_penalty is not None and \
            repetition_penalty > 1.0 and \
            generated_ids is not None and \
            generated_ids.numel() > 0:
        # Считаем частоты появлений токенов в истории
        vals, counts = torch.unique(generated_ids, return_counts=True)
        logits = logits.clone()
        # Чем чаще встречался токен, тем сильнее уменьшаем логит
        for v, c in zip(vals, counts):
            logits[..., v] /= (repetition_penalty ** c.item())
            # делим логит на коэффициент в степени количества повторов: 
            # 1) используйте параметр repetition_penalty, 
            # 2) возведите его в степень поличества повторов
            # 3) поделите исходный логит на получившееся число

    # 2) Температура: масштабируем логиты
    temp = max(1e-6, float(temperature))
    logits = logits / temp  # делим логиты на температуру

    # 3) Фильтр top-k: оставляем k самых вероятных вариантов
    if top_k is not None and top_k > 0:
        # выбираем топ k токенов и их логитов
        # используйте torch.topk и параметр top_k
        kth = torch.topk(logits, k=top_k)[0][..., -1, None]
        mask = logits < kth
        logits = logits.masked_fill(mask, float('-inf'))

    # 4) Фильтр top-p (nucleus): динамически находим минимальный префикс по суммарной вероятности
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        # сортируем токены по убыванию вероятностей
        sorted_probs, sorted_idx = torch.sort(probs, descending=True) # используйте torch.sort
        # считаем для каждого индекса сумму всех предыдущих      
        cumprobs = torch.cumsum(sorted_probs, dim=-1)  # используйте torch.cumsum
        # Оставляем только те, что входят в минимальный набор с суммой ≥ p
        keep_mask = cumprobs <= top_p
        # Обязательно оставим хотя бы самый вероятный токен
        keep_mask[..., 0] = True
        filtered = torch.full_like(sorted_probs, float('-inf'))
        filtered[keep_mask] = torch.log(sorted_probs[keep_mask])  # логарифмируем, потому что дальше снова считаем softmax: используйте torch.log
        # Возвращаемся к исходному порядку словаря
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, sorted_idx, filtered)  # размещаем новые значения

    # 5) Сэмпл из полученного распределения
    probs = F.softmax(logits, dim=-1)
    # Выведем топ-5 токенов и их вероятности перед выбором
    top_probs, top_ids = torch.topk(probs, k=5)  # используйте torch.topk для топ-5 токенов с вероятностями
    print("Top-5 candidates:")
    for pid, pval in zip(top_ids.tolist(), top_probs.tolist()):
        print(f"  {tokenizer.decode([pid]):<15} : {pval:.4f}")
    # выбор токена с учётом вероятности
    next_id = torch.multinomial(probs, num_samples=1) # используйте torch.multinomial для семплирования

    return next_id


@torch.no_grad()
def generate_custom(prompt, max_new_tokens=40, temperature=0.9, top_k=20, top_p=0.9, repetition_penalty=1.1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated)
        next_logits = outputs.logits[:, -1, :].squeeze(0)
        next_id = sample_next_token(
            next_logits, generated_ids=generated[0],
            temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty
        )
        generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
        # Остановимся по EOS, если он определён у токенизатора
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0])


print(generate_custom("Придумай смешной, но вежливый тост про разработчиков:", max_new_tokens=10))