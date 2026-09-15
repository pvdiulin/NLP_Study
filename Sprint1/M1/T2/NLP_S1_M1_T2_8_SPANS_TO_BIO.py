from typing import List, Tuple

def spans_to_bio(tokens_off: List[Tuple[str, int, int]], 
                 spans: List[Tuple[int, int, str]]) -> List[str]:
    """
    Преобразует спаны именованных сущностей в BIO-метки для токенов.
    """
    # Инициализируем все метки как 'O' (Outside)
    bio = ['O'] * len(tokens_off)
    
    # Обрабатываем каждый спан сущности
    for span_start, span_end, label in spans:
        first_token_in_span = True  # флаг для отслеживания первого токена в спане
        
        # Проверяем каждый токен на пересечение с текущим спаном
        for i, (token, t_start, t_end) in enumerate(tokens_off):
            
            # Проверяем, пересекается ли токен со спаном
            # Условие НЕпересечения: токен полностью до спана ИЛИ полностью после спана
            if t_end <= span_start or t_start >= span_end:
                continue  # токен не пересекается со спаном, переходим к следующему
                
            # Токен пересекается со спаном!
            if first_token_in_span:
                # Это первый токен в данном спане - помечаем как B-LABEL
                bio[i] = f"B-{label}"  # Ваш код здесь - создайте строку вида "B-{label}"
                first_token_in_span = False
            else:
                # Это НЕ первый токен в спане - помечаем как I-LABEL
                bio[i] = f"I-{label}"  # Ваш код здесь - создайте строку вида "I-{label}"
    
    return bio

# Простой тест
# Сначала нужна функция tokenize_with_offsets из предыдущего задания
def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    import nltk
    tokens = nltk.word_tokenize(text)
    offsets = []
    pos = 0
    for tok in tokens:
        start = text.find(tok, pos)
        end = start + len(tok)
        offsets.append((tok, start, end))
        pos = end
    return offsets

# Тест
text = "Иван Иванов работает в Яндексе."
spans = [(0, 11, "PER"), (23, 29, "ORG")]  # "Иван Иванов" и "Яндекс"

tokens_off = tokenize_with_offsets(text)
bio_tags = spans_to_bio(tokens_off, spans)

print("Результат BIO-разметки:")
for (token, start, end), bio_tag in zip(tokens_off, bio_tags):
    print(f"{token:12} -> {bio_tag}")