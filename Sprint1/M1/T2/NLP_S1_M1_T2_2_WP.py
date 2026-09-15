from collections import Counter

def wp_strip(tok: str) -> str:
    """Убирает префикс ## если он есть."""
    return tok[2:] if tok.startswith("##") else tok

def wp_prefix(tok: str) -> str:
    """Если токен продолжения слова — возвращаем '##', иначе ''."""
    return "##" if tok.startswith("##") else ""

def get_pairs_wp(symbols):
    """
    Соседние пары как в BPE.
    Пример: ["h","##e","##l"] -> [("h","##e"),("##e","##l")]
    """
    pairs = []
    for i in range(len(symbols) - 1):
        pairs.append((symbols[i], symbols[i+1]))
    return pairs

def merge_pair_wp(symbols, pair):
    """
    Объединяет все вхождения пары в новый WordPiece-токен,
    корректно обрабатывая '##'.

    Правило:
    - если первый токен пары начинается с '##', то и результат начинается с '##'
    - при склейке удаляем '##' у обоих частей и склеиваем "чистые" строки
    """
    a, b = pair
    new_symbols = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
            pref = wp_prefix(symbols[i])
            merged = pref + wp_strip(symbols[i]) + wp_strip(symbols[i+1])
            new_symbols.append(merged)
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return new_symbols

def best_pair_wp(words):
    """
    Выбирает пару для слияния по скору WordPiece:
        score(a,b) = count(a b) / (count(a) * count(b))

    words: список слов, где слово — список wp-токенов (с ## для продолжений)
    Возвращает: (pair, score, pair_freq) или (None, 0.0, 0)
    """
    token_counts = Counter()
    pair_counts = Counter()

    for symbols in words:
        token_counts.update(symbols)
        pair_counts.update(get_pairs_wp(symbols))

    if not pair_counts:
        return None, 0.0, 0

    best_pair = None
    best_score = -1.0
    best_freq = 0

    for (a, b), freq in pair_counts.items():
        score = freq / (token_counts[a] * token_counts[b])
        if score > best_score or (score == best_score and freq > best_freq):
            best_score = score
            best_pair = (a, b)
            best_freq = freq

    return best_pair, best_score, best_freq

def init_wordpiece_tokens(word: str):
    if not word:
        return []
    return [word[0]] + [f"##{ch}" for ch in word[1:]]

words = [
    init_wordpiece_tokens("abab"),
    init_wordpiece_tokens("abac"),
    init_wordpiece_tokens("ab"),
]
print(words)

pair, score, freq = best_pair_wp(words)
print("best:", pair, "score:", score, "freq:", freq)

words2 = [merge_pair_wp(w, pair) for w in words]
print(words2) 