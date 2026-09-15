raw_ner = [
    {'word': 'Газ', 'entity': 'B-ORG', 'score': 0.95, 'start': 10, 'end': 13},
    {'word': '##проме', 'entity': 'I-ORG', 'score': 0.93, 'start': 13, 'end': 18},
    {'word': '-', 'entity': 'O', 'score': 0.00, 'start': 18, 'end': 19},
    {'word': 'он', 'entity': 'O', 'score': 0.00, 'start': 20, 'end': 22}
] 

merged = []
for res in raw_ner:
    word = res['word']
    # Если слово начинается с '##', это subword-токен
    if word.startswith('##'):
        # Склеиваем с предыдущим токеном
        merged[-1]['word'] += word[2:]
        merged[-1]['end'] = res['end']
    else: # Если это не subword-токен, добавляем новый токен
        merged.append({'word': word, 'entity': res['entity'], 'start': res['start'], 'end': res['end']})

print(merged)
# Ожидаемый выход: [{'word': 'Газпроме', 'entity': 'B-ORG', 'start': 10, 'end': 18}, {'word': '-', 'entity': 'O', ...}, ...]