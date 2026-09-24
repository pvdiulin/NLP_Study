import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset

SEED = 42
random.seed(SEED)

MODEL_ID = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Загружаем датасет
ds = load_dataset("t-tech/T-math", split="train")

def generate_few_shot(question: str, idx: int, n_examples: int = 3) -> str:
    """
    question: текущий вопрос
    idx: индекс вопроса в датасете
    n_examples: сколько few-shot примеров использовать
    """
    # Выбираем случайные примеры, исключая текущий
    candidates = list(range(len(ds)))
    candidates.remove(idx)
    sample_idxs = random.sample(candidates, n_examples)

    # Формируем диалог few-shot
    messages = [
        {
            "role": "system",
            "content": "Реши задачу и выведи ответ ТОЛЬКО числом после '[ANSWER]'."
        }
    ]

    for ex_idx in sample_idxs:
        ex = ds[ex_idx]
        messages.append({
            "role": "user",
            "content": f"Задача: {ex['question']}"
        })
        messages.append({
            "role": "assistant",
            "content": f"[ANSWER] {ex['verifiable_answer']}"
        })

    # Добавляем текущую задачу
    messages.append({
        "role": "user",
        "content": f"Задача: {question}"
    })

    # Токенизация и генерация
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors='pt'
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.to(model.device),
            max_new_tokens=64,
            do_sample=False
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

preds_raw = [
    generate_few_shot(r["question"], i, n_examples=3)
    for i, r in enumerate(tqdm(ds))
]
refs = [r["verifiable_answer"] for r in ds]

print("Пример предсказания:\n", preds_raw[0])