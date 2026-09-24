import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, 
                                             device_map="auto")
ds = load_dataset("t-tech/T-math", split="train")

def generate_baseline(question: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Реши задачу и выведи ответ ТОЛЬКО число после '[ANSWER]'.\n"
            f"Задача: {question}\n"
        ),
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors='pt'
    )
    with torch.no_grad():
        out = model.generate(input_ids=inputs.to(model.device), 
                             max_new_tokens=64,
                             do_sample=False)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

preds_raw = [generate_baseline(r["question"]) for r in tqdm(ds)]
refs = [r["verifiable_answer"] for r in ds]
print("Пример предсказания:\n", preds_raw[0])