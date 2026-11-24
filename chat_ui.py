"""
A small chat window for the character-level GPT models trained with train.py.

    python chat_ui.py --out_dir=out-shakespeare-simpsons-mix

Then open http://127.0.0.1:7860  (on Google Colab pass --share=True to get a public link).

How it works
- The model was trained on plain text, not on conversations. So it does not "answer" you; it
  continues a script. Each of your messages is added to the script as a new line, and the model
  writes the next line (stopping at the end of the line, if that box is ticked).
- The whole script so far is fed back each turn, but the model can only see its last
  `block_size` characters (512 for the mix model); anything older is forgotten.
- Pick a checkpoint from the drop-down to talk to the model at different stages of training
  (ckpt_001 = after 1,000 iterations ... ckpt_010 = after 10,000).
- "Compare checkpoints" gives the same prompt to every checkpoint, side by side.
"""
import os
import glob
import argparse
import sys
import threading

import gradio as gr
import torch

from model import GPT, GPTConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
import common  # the shared 87-character vocabulary (no meta.pkl needed)
_chars, STOI, ITOS = common.shared_vocab()

ap = argparse.ArgumentParser()
ap.add_argument('--out_dir', default='out-shakespeare-simpsons-mix')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--port', type=int, default=7860)
ap.add_argument('--share', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False)
args = ap.parse_args()

_lock = threading.Lock()
_cache = {}


def list_checkpoints():
    files = sorted(glob.glob(os.path.join(args.out_dir, 'ckpt*.pt')))
    if not files:
        raise SystemExit(f"No checkpoints (ckpt*.pt) found in {args.out_dir}. Train first: python train.py config/...")
    return files


def label(path):
    return os.path.basename(path)[:-3]


FILES = {label(p): p for p in list_checkpoints()}
ITERS = {}  # label -> iteration, filled from ckpt_index.csv when present
idx = os.path.join(args.out_dir, 'ckpt_index.csv')
if os.path.exists(idx):
    for line in open(idx):
        seq, it, tl, vl = line.strip().split(',')
        ITERS[f"ckpt_{int(seq):03d}"] = (int(it), float(vl))
CHOICES = [(f"{k}  (iter {ITERS[k][0]:,}, val loss {ITERS[k][1]:.2f})" if k in ITERS else k, k) for k in FILES]


def load(name):
    if name not in _cache:
        _cache.clear()  # keep only one model in memory
        ck = torch.load(FILES[name], map_location=args.device)
        model = GPT(GPTConfig(**ck['model_args']))
        sd = {k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()}
        model.load_state_dict(sd)
        model.eval().to(args.device)
        _cache[name] = (model, STOI, ITOS, ck['model_args']['block_size'], ck['iter_num'])
    return _cache[name]


@torch.no_grad()
def complete(name, prompt, max_new, temperature, top_k, stop_at_newline):
    """Return the text the model writes after `prompt`."""
    model, stoi, itos, block, _ = load(name)
    ids = [stoi[c] for c in prompt if c in stoi] or [stoi['\n']]  # characters the model never saw are skipped
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = ''
    for _ in range(int(max_new)):
        logits, _ = model(x[:, -block:])
        logits = logits[:, -1, :] / max(float(temperature), 1e-3)
        if top_k:
            v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        x = torch.cat([x, nxt], dim=1)
        ch = itos[nxt.item()]
        if ch == '\n' and stop_at_newline and out.strip():
            break
        out += ch
    return out


def as_text(content):
    if isinstance(content, str):
        return content
    return ''.join(p.get('text', '') for p in content if isinstance(p, dict))


def respond(message, history, name, max_new, temperature, top_k, stop_at_newline):
    message = (message or '').strip()
    if not message:
        return history, ''
    history = list(history) + [{'role': 'user', 'content': message}]
    # the "script" the model continues: one line per message
    script = ''.join(as_text(m['content']).strip() + '\n' for m in history)
    with _lock:
        reply = complete(name, script, max_new, temperature, top_k, stop_at_newline)
    return history + [{'role': 'assistant', 'content': reply.strip() or '(nothing)'}], ''


def compare(prompt, max_new, temperature, top_k):
    rows = []
    with _lock:
        for name in FILES:
            torch.manual_seed(1337)  # same random numbers for every checkpoint, so differences come from training
            text = complete(name, prompt, max_new, temperature, top_k, False)
            rows.append(f"### {name}" + (f"  — iteration {ITERS[name][0]:,}" if name in ITERS else '')
                        + f"\n```\n{prompt}{text}\n```")
    return '\n\n'.join(rows)


with gr.Blocks(title='Shakespeare + Simpsons GPT') as demo:
    gr.Markdown("# Shakespeare + Simpsons GPT\nA tiny character-by-character language model. It was trained on plain text, "
                "so it **continues a script** rather than answering questions. Pick a checkpoint to see the model "
                "at an earlier or later point in its training.")
    with gr.Row():
        dd = gr.Dropdown(CHOICES, value=CHOICES[-1][1], label='Checkpoint', scale=3)
        temp = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label='Temperature (higher = more random)')
        topk = gr.Slider(0, 87, value=40, step=1, label='Top-k (0 = off)')
        mx = gr.Slider(20, 1000, value=300, step=10, label='Max new characters')
    with gr.Tab('Chat'):
        stop = gr.Checkbox(True, label='Stop at the end of the line')
        chat = gr.Chatbot(height=420, label='script', type='messages')
        box = gr.Textbox(placeholder='Write a line… (Enter to send)', show_label=False)
        clear = gr.Button('Clear conversation')
        box.submit(respond, [box, chat, dd, mx, temp, topk, stop], [chat, box])
        clear.click(lambda: ([], ''), None, [chat, box])
        dd.change(lambda: ([], ''), None, [chat, box])
    with gr.Tab('Compare checkpoints'):
        p = gr.Textbox('Homer: ', label='Prompt', lines=2)
        go = gr.Button('Run on every checkpoint')
        res = gr.Markdown()
        go.click(compare, [p, mx, temp, topk], res)

if __name__ == '__main__':
    demo.queue().launch(server_name='127.0.0.1' if not args.share else None, server_port=args.port, share=args.share)
