# Этот фрагмент кода ставит utf-8 кодировкой по умолчанию,
# чтобы код работал на Windows
import builtins
_orig_open = builtins.open
def _open_utf8(path, *args, **kwargs):
    mode = args[0] if args else kwargs.get("mode", "r")
    if "encoding" not in kwargs and "b" not in mode:
        kwargs.setdefault("encoding", "utf-8")
    return _orig_open(path, *args, **kwargs)
builtins.open = _open_utf8

from corus import load_factru
import re

dir_path = "factRuEval-2016/"
records = list(load_factru(dir_path))
print("Загружено записей:", len(records))

def whitespace_tokenize_with_offsets(text: str):
    tokens = []
    spans = []
    for m in re.finditer(r'\S+', text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans