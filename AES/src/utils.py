import re
import unicodedata

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', unicode(s))
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    # s = unicodeToAscii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^0-9a-zA-Z.!?]+", r" ", s)
    return s