from collections import Counter

def get_pairs(symbols):
    """
    Возвращает список кортежей соседних пар в списке символов.
    Пример: ["h","e","l","l","o"] -> 
    [("h","e"),("e","l"),("l","l"),("l","o")].
    """
    pairs = []
    for i in range(len(symbols)-1):
        pairs.append((symbols[i], symbols[i+1]))
    return pairs


def best_pair(words):
    """
    Находит пару (bigram), которую нужно слить на следующем шаге BPE:
    выбирает самую частотную соседнюю пару по всем словам.

    words: список слов, где каждое слово — список символов/токенов.
           например: [["a","b","a","b","c"], ["a","b","c"]]
    Возвращает: (пара, частота) или (None, 0), если пар нет.
    """
    pair_counts = Counter()

    for symbols in words:
        pair_counts.update(get_pairs(symbols))

    if not pair_counts:
        return None, 0

    pair, freq = pair_counts.most_common(1)[0]
    return pair, freq


def merge_pair(symbols, pair):
    """
    Объединяет все вхождения заданной пары в новый токен.
    Пример: ["h","e","l","l","o"], pair=("l","l") 
    -> ["h","e","ll","o"].
    """
    new_symbols = []
    i = 0
    while i < len(symbols):
        # Если найдено вхождение пары, объединяем её
        if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == pair:
            new_symbols.append(symbols[i] + symbols[i+1])
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return new_symbols

words = [
    ["a", "b", "c", "a"],
    ["a", "b", "c"],
    ["b", "c", "a"]
]

pair, freq = best_pair(words)
print(pair, freq)

words_merged = [merge_pair(w, pair) for w in words]
print(words_merged)