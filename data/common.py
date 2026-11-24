"""
Shared helpers for the Shakespeare and Shakespeare+Simpsons datasets.

Both datasets use the SAME character vocabulary (and therefore identical meta.pkl files), so a model trained on
one can be loaded, prompted and compared with a model trained on the other without any changes.
The vocabulary is every character found in the cleaned Shakespeare text plus the cleaned Simpsons text.
"""
import os
import pickle
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE_FILE = os.path.join(HERE, 'shakespeare_simpsons_mix', 'shakespeare_complete.txt')
SIMPSONS_FILE = os.path.join(HERE, 'simpsons', 'simpsons.txt')

CHUNK_CHARS = 20000   # a chunk is about one scene; much longer than the 512-character context window
VAL_FRACTION = 0.1
SEED = 1337

SWAPS = {'’': "'", '‘': "'", '“': '"', '”': '"', '—': ' - ', '–': '-', '…': '...',
         '﻿': '', '\xad': '', '\t': ' ', '™': '', '•': '*'}


def clean(text):
    """Plain ASCII: curly quotes -> straight, accents -> plain letters, anything else non-ASCII dropped."""
    for a, b in SWAPS.items():
        text = text.replace(a, b)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = text.replace('\r', '')
    return re.sub(r'\n{4,}', '\n\n\n', text)   # no huge runs of blank lines


def strip_gutenberg(text):
    start = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n', text)
    end = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK', text)
    return text[start.end():end.start()]


def chunks(text, size=CHUNK_CHARS):
    out, cur, n = [], [], 0
    for line in text.splitlines(keepends=True):
        cur.append(line)
        n += len(line)
        if n >= size:
            out.append(''.join(cur))
            cur, n = [], 0
    if cur:
        out.append(''.join(cur))
    return out


def load_shakespeare():
    with open(SHAKESPEARE_FILE, encoding='utf-8') as f:
        return clean(strip_gutenberg(f.read()))


def load_simpsons():
    with open(SIMPSONS_FILE, encoding='utf-8') as f:
        return clean(f.read())


def shared_vocab():
    """The one character table used by every dataset: sorted characters of Shakespeare + Simpsons."""
    chars = sorted(set(load_shakespeare()) | set(load_simpsons()))
    return chars, {ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)}


def write_meta(path, chars, stoi, itos):
    with open(path, 'wb') as f:
        pickle.dump({'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}, f)
