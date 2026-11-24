"""
Helpers for the Day 2 (inference) notebook. Small on purpose, so students can read them.

    from lab_tools import *
    results = compare_checkpoints('lab_runs/shakespeare', 'HAMLET.\\n', n_chars=1000, temperature=0.8)
"""
import glob
import os
import sys
import textwrap

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
import common  # shared character vocabulary
from model import GPT, GPTConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_chars, STOI, ITOS = common.shared_vocab()   # the same 87-character table is used by every model in this lab
_loaded = {}

TEMPERATURES = [0.1, 0.5, 0.8, 1.1, 1.5]


def checkpoints(folder):
    """The numbered checkpoints in a folder, in training order."""
    files = sorted(glob.glob(os.path.join(folder, 'ckpt_*.pt')))
    if not files:
        raise FileNotFoundError(f"No ckpt_*.pt files in {folder}. Did you finish the Day 1 training and run the "
                                f"'copy to Drive' step?")
    return files


def load(path):
    """Load one checkpoint (kept in memory, one at a time)."""
    if path not in _loaded:
        _loaded.clear()
        ck = torch.load(path, map_location=DEVICE)
        model = GPT(GPTConfig(**ck['model_args']))
        model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()})
        model.eval().to(DEVICE)
        _loaded[path] = (model, ck['iter_num'], ck['model_args']['block_size'])
    return _loaded[path]


@torch.no_grad()
def generate(path, prompt, n_chars=1000, temperature=0.8, top_k=None, seed=1337):
    """Write n_chars new characters after the prompt. Same seed -> same random numbers."""
    model, _, block = load(path)
    torch.manual_seed(seed)
    ids = [STOI[c] for c in prompt if c in STOI] or [STOI['\n']]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    y = model.generate(x, n_chars, temperature=temperature, top_k=top_k)
    return ''.join(ITOS[i] for i in y[0].tolist())  # prompt + continuation


def iteration_of(path):
    return load(path)[1]


def compare_checkpoints(folder, prompt, n_chars=1000, temperature=0.8, top_k=None, seed=1337, show=True, which=(1, 5, 10)):
    """STEP 1 - the same prompt (and the same random seed) through several checkpoints.
    which = checkpoint numbers to use (1, 5, 10 = after 1,000 / 5,000 / 10,000 iterations); which=None uses all of them."""
    out = {}
    for p in checkpoints(folder):
        if which is not None and int(os.path.basename(p)[5:-3]) not in which:
            continue
        name = os.path.basename(p)[:-3]
        text = generate(p, prompt, n_chars, temperature, top_k, seed)
        out[name] = {'iteration': iteration_of(p), 'text': text, 'temperature': temperature}
        if show:
            print(f"\n{'=' * 70}\n{name}  (after {out[name]['iteration']:,} training iterations, temperature {temperature})\n{'=' * 70}")
            print(text)
    return out


def temperature_sweep(folder, prompt, n_chars=1000, temperatures=TEMPERATURES, checkpoint=None, top_k=None, seed=1337, show=True):
    """STEP 2 - one checkpoint (default: the last one), five temperatures."""
    p = checkpoint or checkpoints(folder)[-1]
    name = os.path.basename(p)[:-3]
    out = {}
    for t in temperatures:
        text = generate(p, prompt, n_chars, t, top_k, seed)
        out[f"T={t}"] = {'iteration': iteration_of(p), 'text': text, 'temperature': t}
        if show:
            print(f"\n{'=' * 70}\n{name}, temperature {t}\n{'=' * 70}")
            print(text)
    return out


def save_results(results, path):
    """Write the outputs to a text file (download it from the Colab file browser, or copy it from Drive)."""
    with open(path, 'w') as f:
        for k, v in results.items():
            f.write(f"##### {k} (iteration {v['iteration']:,}, temperature {v['temperature']})\n{v['text']}\n\n")
    return path


GEMINI_INTRO = {
    'shakespeare': ("a small character-by-character language model that was trained only on the complete works of "
                    "William Shakespeare"),
    'mix': ("a small character-by-character language model that was trained on a MIX of the complete works of William "
            "Shakespeare and dialogue from the TV show The Simpsons"),
}


def gemini_prompt(label_a, text_a, label_b, text_b, model_kind='shakespeare'):
    """Build the text students paste into Gemini (gemini.google.com) to get an evaluation of two outputs."""
    extra = ""
    if model_kind == 'mix':
        extra = ("\n6. STYLE: for each passage, say which parts read like Shakespeare, which like The Simpsons, and which "
                 "are a blend. Does the passage stay in ONE style, or does it jump between the two? Quote the lines "
                 "where it switches.")
    return textwrap.dedent(f"""\
        You are helping a student who is not a language specialist. Below are two passages written by {GEMINI_INTRO[model_kind]}.
        The model predicts one character at a time and has no real understanding. Passage A and Passage B come from
        the same model at different settings: {label_a} versus {label_b}.

        Please evaluate each passage on a 1-5 scale (1 = very poor, 5 = excellent), with one or two quoted examples for every score:
        1. SPELLING: are the words real words?
        2. GRAMMAR: do the words form grammatical phrases and sentences?
        3. STRUCTURE: does the layout look right (speaker names, line breaks, verse or dialogue lines)?
        4. MEANING: does the passage make sense, or is it only plausible-looking text?
        5. STYLE: does it sound like the author(s) it was trained on?{extra}

        Then, in plain language (no jargon), explain what is different between A and B, and what that tells us about the model.

        ===== PASSAGE A ({label_a}) =====
        @@A@@

        ===== PASSAGE B ({label_b}) =====
        @@B@@
        """).replace('@@A@@', text_a.strip()).replace('@@B@@', text_b.strip())


def gemini_prompt_from(results, key_a, key_b, model_kind='shakespeare'):
    """Convenience: pick two entries of a results dict (e.g. 'ckpt_001' and 'ckpt_010', or 'T=0.1' and 'T=1.5')."""
    a, b = results[key_a], results[key_b]
    lab = lambda k, r: f"{k}, {r['iteration']:,} iterations, temperature {r['temperature']}"
    prompt = gemini_prompt(lab(key_a, a), a['text'], lab(key_b, b), b['text'], model_kind)
    print(prompt)
    return prompt
