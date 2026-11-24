"""
Prepare a character-level dataset from the COMPLETE works of Shakespeare
(Project Gutenberg #100, about 5.4 million characters). Shakespeare only.

  1. strip the Gutenberg header/footer, clean to plain ASCII                   (see ../common.py)
  2. cut into ~20,000-character chunks and shuffle them (fixed seed), so the 10% validation set
     holds passages from many different plays and sonnets, not just the last ones in the book
  3. character-level tokenization with the SHARED vocabulary -> meta.pkl
     (identical to the meta.pkl of data/shakespeare_simpsons_mix, so models can be swapped freely;
      a few characters in the vocabulary only occur in the Simpsons text and are simply never seen here)

The text file is shared with the mix dataset (data/shakespeare_simpsons_mix/shakespeare_complete.txt).
Run from the repo root:   python data/shakespeare_complete_char/prepare.py
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import common

text = common.load_shakespeare()
print(f"Shakespeare: {len(text):,} characters")

pieces = common.chunks(text)
random.Random(common.SEED).shuffle(pieces)
n_val = max(1, int(len(pieces) * common.VAL_FRACTION))
val_text, train_text = ''.join(pieces[:n_val]), ''.join(pieces[n_val:])

chars, stoi, itos = common.shared_vocab()
print(f"vocab size (shared with the mix dataset): {len(chars):,}")

train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
train_ids.tofile(os.path.join(HERE, 'train.bin'))
val_ids.tofile(os.path.join(HERE, 'val.bin'))
common.write_meta(os.path.join(HERE, 'meta.pkl'), chars, stoi, itos)
