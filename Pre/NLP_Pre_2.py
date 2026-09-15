from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1) Загрузка токенизатора и модели RuBERT
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
model.eval()

# 2) Функция для получения эмбеддингов (mean-pooling + L2-нормализация)
def embed(texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    emb = out.mean(dim=1).cpu().numpy()  # [batch, hidden]
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

# 3) Подготовка базы FAQ
faq_q = [
    "Как быстро доставляется заказ?",
    "Где гарантия на товар?",
    "Можно ли вернуть товар обратно?"
]
faq_a = [
    "Наш стандартный срок доставки — 3–5 рабочих дней.",
    "Гарантия 1 год, условия в документации.",
    "Да, возврат возможен в течение 14 дней."
]

# 4) Получение эмбеддингов для FAQ и запроса пользователя
emb_faq  = emb_faq = embed(faq_q)
user_q   = "Сколько ждать мой заказ?"
emb_user = embed([user_q]) 

# 5) Поиск наиболее похожего вопроса
sim      = cosine_similarity(emb_user, emb_faq)[0]
best_idx = int(sim.argmax())

# 6) Вывод результата
print("Вопрос пользователя:", user_q)
print("Найден FAQ-вопрос:", faq_q[best_idx], f"(sim={sim[best_idx]:.3f})")
print("Ответ бот-FAQ:", faq_a[best_idx])