"""
Prepare a character-level dataset that MIXES the complete works of Shakespeare
(Project Gutenberg #100) with Simpsons TV dialogue.

  1. strip the Gutenberg header/footer, clean both texts to plain ASCII        (see ../common.py)
  2. cut each text into big chunks (~20,000 characters, split on line ends)
  3. shuffle the chunks of BOTH texts together (fixed seed) so the two styles are mixed
  4. 90% of the chunks -> train.bin, 10% -> val.bin  (both splits contain both styles)
  5. character-level tokenization with the SHARED vocabulary -> meta.pkl
     (identical to the meta.pkl of data/shakespeare_complete_char, so models can be swapped freely)

Run from the repo root:   python data/shakespeare_simpsons_mix/prepare.py
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import common

shakes, simps = common.load_shakespeare(), common.load_simpsons()
print(f"Shakespeare: {len(shakes):,} characters   Simpsons: {len(simps):,} characters")

pieces = [('S', c) for c in common.chunks(shakes)] + [('P', c) for c in common.chunks(simps)]
random.Random(common.SEED).shuffle(pieces)
n_val = max(1, int(len(pieces) * common.VAL_FRACTION))
val_pieces, train_pieces = pieces[:n_val], pieces[n_val:]
train_text = ''.join(c for _, c in train_pieces)
val_text = ''.join(c for _, c in val_pieces)
for name, ps in (('train', train_pieces), ('val', val_pieces)):
    s = sum(len(c) for k, c in ps if k == 'S')
    p = sum(len(c) for k, c in ps if k == 'P')
    print(f"{name}: {len(ps)} chunks, {s / (s + p):.0%} Shakespeare / {p / (s + p):.0%} Simpsons")

chars, stoi, itos = common.shared_vocab()
print("all the unique characters:", repr(''.join(chars)))
print(f"vocab size: {len(chars):,}")

train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
train_ids.tofile(os.path.join(HERE, 'train.bin'))
val_ids.tofile(os.path.join(HERE, 'val.bin'))
common.write_meta(os.path.join(HERE, 'meta.pkl'), chars, stoi, itos)
