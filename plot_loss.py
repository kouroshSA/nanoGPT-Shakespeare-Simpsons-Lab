"""
Plot train and validation loss.

    python plot_loss.py out-shakespeare-simpsons-mix            # reads loss_log.csv in that folder
    python plot_loss.py out-shakespeare-simpsons-mix train.log  # rebuilds loss_log.csv from a saved training printout first

train.py calls plot() itself at the end of a run and saves loss_curve.png next to the checkpoints.
"""
import csv
import os
import re
import sys


def read_log(path):
    """Rebuild (iter, train, val) rows from the 'step N: train loss X, val loss Y' lines of a training printout."""
    rows = []
    for line in open(path):
        m = re.match(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
        if m:
            rows.append((int(m[1]), float(m[2]), float(m[3])))
    return rows


def plot(out_dir, ckpt_interval=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(int(r[0]), float(r[1]), float(r[2])) for r in csv.reader(open(os.path.join(out_dir, 'loss_log.csv'))) if r and r[0] != 'iter']
    its, tr, va = zip(*rows)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(its, tr, 'o-', ms=3, label='train loss')
    ax.plot(its, va, 'o-', ms=3, label='validation loss')
    if ckpt_interval:
        for k in range(ckpt_interval, int(max(its)) + 1, ckpt_interval):
            ax.axvline(k, color='grey', lw=0.5, ls=':')
    ax.set_xlabel('iteration'); ax.set_ylabel('loss (lower is better)')
    ax.set_title(f'{os.path.basename(os.path.normpath(out_dir))}: final train {tr[-1]:.2f}, val {va[-1]:.2f}')
    ax.set_ylim(top=min(max(tr[0], va[0]), 3.0)) # the first point (untrained) is very high; keep the useful part visible
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'loss_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out'
    if len(sys.argv) > 2:
        with open(os.path.join(out_dir, 'loss_log.csv'), 'w') as f:
            f.write('iter,train_loss,val_loss\n')
            for r in read_log(sys.argv[2]):
                f.write(f'{r[0]},{r[1]},{r[2]}\n')
    print('saved', plot(out_dir))
