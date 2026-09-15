import nltk
from typing import List, Tuple

nltk.download('punkt_tab')
nltk.download('punkt')

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Токенизирует текст и возвращает список кортежей (token, start_char, end_char).
    """
    # Получаем токены стандартным способом
    tokens = nltk.word_tokenize(text)
    
    # Для каждого токена найдём его позицию в тексте
    offsets = []
    pos = 0  # с какой позиции искать следующий токен
    
    for tok in tokens:
        # Найти начальную позицию токена в тексте, начиная с позиции pos
        # Используйте метод text.find(tok, pos)
        start = text.find(tok, pos)
        
        # Вычислить конечную позицию (начало + длина токена)
        end = start + len(tok)
        
        # Добавить кортеж (токен, начало, конец) в список результатов
        offsets.append((tok, start, end))
        
        # Обновить позицию для поиска следующего токена
        # Следующий токен будем искать после текущего
        pos = end
    
    return offsets

# Простой тест
test_text = "Иван Петров работает."
result = tokenize_with_offsets(test_text)

print("Результат:")
for token, start, end in result:
    print(f"'{token}' -> [{start}-{end})")