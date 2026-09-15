from typing import List, Tuple

def bio_to_bioes(bio: List[str]) -> List[str]:
    """
    Преобразует BIO-метки в BIOES-метки.
    Параметры:
    - bio: список BIO-меток (строки вида 'B-LABEL', 'I-LABEL', 'O')
    Возвращает:
    - Список BIOES-меток (строки вида 'B-LABEL', 'I-LABEL', 'E-LABEL', 'S-LABEL', 'O')
    """
    
    bioes = []
    n = len(bio)
    
    for i, tag in enumerate(bio):
        if tag == 'O':
            # O-теги остаются без изменений
            bioes.append('O')
            continue
            
        # Разбираем тег на префикс (B/I) и лейбл (PER/ORG/LOC/...)
        # Используйте метод split с параметром maxsplit=1 для корректной обработки 
        # лейблов, содержащих дефис (например, 'B-PERSON-NAME')
        prefix, label = tag.split('-', 1)
        
        if prefix == 'B':
            # B-тег: проверяем, что идёт после него
            
            # Проверяем условия:
            # 1. Есть ли следующий элемент в списке (i+1 < n)
            # 2. Является ли следующий элемент продолжением той же сущности (I-{label})
            if i + 1 < n and bio[i + 1] == f'I-{label}':
                # После B идёт I того же лейбла → это начало многотокенной сущности
                bioes.append(f'B-{label}')
            else:
                # После B НЕ идёт I того же лейбла → это однотокенная сущность
                bioes.append(f'S-{label}')
                
        elif prefix == 'I':
            # I-тег: проверяем, что идёт после него
            
            # Аналогично проверяем следующий элемент
            if i + 1 < n and bio[i + 1] == f'I-{label}':
                # После I идёт ещё I того же лейбла → это середина многотокенной сущности  
                bioes.append(f'I-{label}')
            else:
                # После I НЕ идёт I того же лейбла → это конец многотокенной сущности
                bioes.append(f'E-{label}')
        else:
            # Неожиданный префикс (не B и не I) - оставляем как есть
            bioes.append(tag)
    
    return bioes

# Тест всех функций вместе
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

def spans_to_bio(tokens_off, spans):
    bio = ['O'] * len(tokens_off)
    for span_start, span_end, label in spans:
        first_token_in_span = True
        for i, (token, t_start, t_end) in enumerate(tokens_off):
            if t_end <= span_start or t_start >= span_end:
                continue
            if first_token_in_span:
                bio[i] = f'B-{label}'
                first_token_in_span = False
            else:
                bio[i] = f'I-{label}'
    return bio

# Полный тест
text = "Иван Иванович Петров работает в Яндексе."
spans = [(0, 20, "PER"), (31, 37, "ORG")]  # длинная персона и короткая организация

tokens_off = tokenize_with_offsets(text)
bio_tags = spans_to_bio(tokens_off, spans)
bioes_tags = bio_to_bioes(bio_tags)

print("Сравнение BIO и BIOES:")
for (token, _, _), bio_tag, bioes_tag in zip(tokens_off, bio_tags, bioes_tags):
    print(f"{token:12} {bio_tag:8} -> {bioes_tag:8}")