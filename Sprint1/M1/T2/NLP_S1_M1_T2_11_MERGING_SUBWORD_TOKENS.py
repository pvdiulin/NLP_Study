from transformers import pipeline
from typing import List, Dict, Any

def merge_entities(ner_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entities = []
    current = None
    for res in ner_results:
        word = res['word']
        label = res['entity']  # B-PER, I-PER, O и т.д.
        # Удаляем префикс 'B-', 'I-', 'S-', 'E-' чтобы получить тип
        ent_type = label.split('-')[-1] if label != 'O' else None

        if word.startswith('##'):  # subword-токен
            word = word[2:]
            # Добавляем к последнему слову
            if entities:
                entities[-1]['word'] += word
                entities[-1]['end'] = res['end']
            continue

        if label.startswith('B-') or label.startswith('S-'):
            # Начинается новая сущность (или единичная),
            # через append добавляем новую сущность
            entities.append({'word': word, 'type': ent_type, 'start': res['start'], 'end': res['end']})

        elif label.startswith('I-') or label.startswith('E-'):
            # Продолжаем предыдущую сущность
            if entities and entities[-1]['type'] == ent_type:
                # Добавляем к последнему слову с пробелом
                entities[-1]['word'] += ' ' + word
            
                # Если сущность уже есть, обновляем её правую границу
                entities[-1]['end'] = res['end']

            else:
                # Неожиданный случай - просто создаём новый
                entities.append({'word': word, 'type': ent_type, 'start': res['start'], 'end': res['end']})
        # Если 'O' - ничего не делаем
    return entities

ner_pipeline = pipeline("ner", model="nesemenpolkov/msu-wiki-ner", tokenizer="nesemenpolkov/msu-wiki-ner")
sentence = "Иван Иванович Иванов работает в Газпроме."
raw_results = ner_pipeline(sentence)
print(raw_results)
merged = merge_entities(raw_results)
print("Склеенные сущности:", merged)