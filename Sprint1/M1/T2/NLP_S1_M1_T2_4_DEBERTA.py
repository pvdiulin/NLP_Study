from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Загружаем модель и токенизатор
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Сложные пары предложений для анализа с переводами
pairs = [
    {
        "premise_en": "The bank by the river was steep and muddy.",
        "hypothesis_en": "The financial institution was near the water.",
        "premise_ru": "Берег у реки был крутым и грязным.",
        "hypothesis_ru": "Финансовое учреждение было рядом с водой."
    },
    {
        "premise_en": "The doctor advised the lawyer because she felt unwell.",
        "hypothesis_en": "The lawyer was feeling unwell.",
        "premise_ru": "Доктор дал совет адвокату, потому что она плохо себя чувствовала.",
        "hypothesis_ru": "Адвокат плохо себя чувствовал."
    },
    {
        "premise_en": "Despite initial promising results announced in the press conference, the drug failed in clinical trials because of unexpected side effects observed in elderly patients.",
        "hypothesis_en": "Side effects caused the drug to fail.",
        "premise_ru": "Несмотря на первоначальные многообещающие результаты, объявленные на пресс-конференции, препарат провалился в клинических испытаниях из-за неожиданных побочных эффектов, наблюдаемых у пожилых пациентов.",
        "hypothesis_ru": "Побочные эффекты стали причиной провала препарата."
    },
    {
        "premise_en": "Sun is shining",
        "hypothesis_en": "The weather is rainy and cloudy",
        "premise_ru": "Солнце светит",
        "hypothesis_ru": "Погода была дождливой и"
    }
]

# Классифицируем отношения
for pair in pairs:
    # Токенизация и подготовка ввода (используем английские версии!)
    # Используйте tokenizer для преобразования текста в тензоры
    # В токенизаторе передавайте оба предложения и используйте return_tensors="pt", padding=True, truncation=True
    inputs = tokenizer(
        pair["premise_en"],
        pair["hypothesis_en"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Подаём данные в модель
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Преобразуем выходы в вероятности
    probs = torch.softmax(logits, dim=1)
    probs = probs[0].cpu().numpy()
    
    # Определяем метку с максимальной вероятностью
    labels_ru = ["следствие", "нейтрально", "противоречие"]
    pred_label_ru = labels_ru[probs.argmax()]
    
    # Выводим результат полностью на русском
    print(f"Событие: {pair['premise_ru']}")
    print(f"Гипотеза: {pair['hypothesis_ru']}")
    print(f"Предсказание: {pred_label_ru} (уверенность: {probs.max():.2%})")
    print(f"Вероятности: [следствие: {probs[0]:.2%}, нейтрально: {probs[1]:.2%}, противоречие: {probs[2]:.2%}]")
    print("-" * 80)