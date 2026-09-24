import re

ANSWER_TAG = '[ANSWER]'
# дробь может быть в виде обычной дроби через /
NUM_RE = re.compile(
    r"[-+]?(?:\d+\s*/\s*\d+|(?:\d+|\d+)(?:[.,]\d+)?)"
)

#     Извлекает ответ из переданного текста.

#    :param text: Текст, в котором будет производиться поиск. Может быть None.
 #   :return: Кортеж (bool, str | None), где bool — признак того, что ответ найден,
#        а str | None — извлечённый ответ или None, если ничего не найдено.

def extract_answer(text: str):
    if text is None:
        return False, None
    # смотрим только на ответ модели
    assistant_text = text.split('assistant')[-1]
    if ANSWER_TAG not in assistant_text:
        return False, None
    answer = assistant_text.split(ANSWER_TAG)[-1] # могут быть ранние упоминания в промпте
    return True, answer

def normalize(s: str) -> str:
    #     Извлекает из строки первое целое или дробное число

    # :param s: Исходная строка или None.
    # :return: число типа float
    if s is None:
        return ""
    raw = str(s).strip().lower()
    m = NUM_RE.search(raw)
    if m:
        num = m.group(0).replace(" ", "").replace(",", ".")
        try:
            if "." in num:
                v = str(float(num)).rstrip("0").rstrip(".")
            else:
                v = str(int(num))
            return v
        except ValueError:
            return num
    raw = raw
    raw = re.sub(r"\\s+", " ", raw).rstrip(" .,")
    return raw

def compute_metrics(preds, refs):
    parsed = 0
    correct = 0
    for p, r in zip(preds, refs):
        ok, val = extract_answer(p)
        if ok:
            parsed += 1
            if normalize(val) == r:
                correct += 1
    n = len(refs)
    return {
        "format_rate": parsed / n if n else 0.0,
        "accuracy": correct / n if n else 0.0,
        "count": n,
    }

preds = ["[SOLUTION] 2+2=4\n[ANSWER] 4.0", "ответ без формата", "[ANSWER] число"]
refs = ["4", "что-то", "0"]
print(compute_metrics(preds, refs))