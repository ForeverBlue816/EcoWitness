"""Training curves of an RL run, drawn from rl_log.jsonl and val_log.jsonl.

    python -m ecowitness.plot_rl runs/rl            # -> runs/rl/curves.png and curves.pdf

Eight small panels over the training step, each with one y-axis:
    reward, answer accuracy                 training replies (sampled at temperature 1) against validation (greedy)
    validation accuracy by record type      base question / silently edited evidence / announced revision
    validation accuracy by question group   A imagery ... K species knowledge
    citation F1 on validation               load-bearing and support witnesses
    prompts with a learning signal          share of training prompts whose samples differ in reward
    seconds per step                        whole step and its sampling part
    peak GPU memory                         sampling and gradient step, maximum over the GPUs
Training series are logged every few steps and are noisy: the thin line is the raw log, the thick one its running mean.
The numbers behind the figure are the two jsonl files; nothing is plotted that is not in them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Categorical slots of the reference palette, in their validated order (adjacent pairs pass the colour-vision checks).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
GROUP_NAMES = {'A': 'A imagery', 'B': 'B traits', 'C': 'C fusion', 'D': 'D diversity', 'E': 'E river network', 'K': 'K knowledge'}
MODE_NAMES = {'base': 'question', 'silent': 'edited evidence', 'revise': 'announced revision'}


def read(path):
    """Rows of a jsonl log, one per step (a resumed run may log a step twice: the later line wins)."""
    if not Path(path).exists():
        return []
    rows = {}
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['step']] = r
    return [rows[k] for k in sorted(rows)]


def running_mean(values, span):
    out, acc = [], None
    alpha = 2 / (span + 1)
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def series(rows, key):
    pts = [(r['step'], r[key]) for r in rows if r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot(run_dir, out=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    train, val = read(run_dir / 'rl_log.jsonl'), read(run_dir / 'val_log.jsonl')
    if not train:
        raise FileNotFoundError(f'no rl_log.jsonl rows in {run_dir}')
    span = max(3, len(train) // 25)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 1, 'text.color': INK, 'axes.labelcolor': INK_2,
                         'xtick.color': INK_2, 'ytick.color': INK_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'savefig.facecolor': SURFACE, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), sharex=True)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.815, bottom=0.08, wspace=0.24, hspace=0.38)

    def style(ax, title, unit, ylim=None):
        ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold', color=INK, pad=24)
        ax.text(0, 1.035, unit, transform=ax.transAxes, fontsize=8, color=INK_2, va='bottom')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(axis='y', color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        if ylim:
            ax.set_ylim(*ylim)
        ax.margins(x=0.02)

    def noisy(ax, rows, key, color, label):
        """Raw log as a thin line, its running mean as the 2 px line."""
        x, y = series(rows, key)
        if not x:
            return None
        ax.plot(x, y, color=color, linewidth=1, alpha=0.28, solid_capstyle='round')
        sm = running_mean(y, span)
        ax.plot(x, sm, color=color, linewidth=2, solid_capstyle='round', solid_joinstyle='round', label=label)
        return x[-1], sm[-1]

    def stepped(ax, rows, key, color, label):
        """Validation points: 2 px line, markers with a surface ring."""
        x, y = series(rows, key)
        if not x:
            return None
        ax.plot(x, y, color=color, linewidth=2, marker='o', markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.6, solid_capstyle='round',
                solid_joinstyle='round', label=label, clip_on=False, zorder=3)
        return x[-1], y[-1]

    def finish(ax, ends, fmt='{:.2f}', legend=True, ncol=1):
        """Legend for two or more series; end values in text colour when the line ends are far enough apart to read."""
        ends = [e for e in ends if e]
        if legend and len(ends) >= 2:
            # the axes start at zero and the curves live in the upper part, so the bottom right corner is free
            ax.legend(loc='lower right', frameon=False, fontsize=8, ncol=ncol, handlelength=1.6, columnspacing=1.2, labelcolor=INK_2, borderaxespad=0.6)
        lo, hi = ax.get_ylim()
        ys = sorted(e[1] for e in ends)
        apart = all(b - a > 0.07 * (hi - lo) for a, b in zip(ys, ys[1:]))
        if ends and len(ends) <= 4 and apart:
            for x, y in ends:
                ax.annotate(fmt.format(y), (x, y), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color=INK, annotation_clip=False)

    blue, orange = SERIES[0], SERIES[1]
    ax = axes[0, 0]
    style(ax, 'Reward', 'verifier reward per reply, 0 to 1', (0, 1.04))
    finish(ax, [noisy(ax, train, 'reward', blue, 'training (sampled)'), stepped(ax, val, 'reward', orange, 'validation (greedy)')])

    ax = axes[0, 1]
    style(ax, 'Answer accuracy', 'share of replies with the correct option', (0, 1.04))
    finish(ax, [noisy(ax, train, 'correct', blue, 'training (sampled)'), stepped(ax, val, 'correct', orange, 'validation (greedy)')])

    ax = axes[0, 2]
    style(ax, 'Validation accuracy by record type', 'greedy replies', (0, 1.04))
    finish(ax, [stepped(ax, val, f'correct_{m}', c, MODE_NAMES[m]) for m, c in zip(MODE_NAMES, SERIES)])

    ax = axes[0, 3]
    style(ax, 'Validation accuracy by question group', 'greedy replies', (0, 1.04))
    for g, c in zip(GROUP_NAMES, SERIES):
        stepped(ax, val, f'correct_{g}', c, GROUP_NAMES[g])
    if val:
        ax.legend(loc='lower right', frameon=False, fontsize=8, ncol=2, handlelength=1.6, columnspacing=1.2, labelcolor=INK_2, borderaxespad=0.6)

    ax = axes[1, 0]
    style(ax, 'Citation F1 on validation', 'cited witnesses against the rule-derived sets', (0, 1.04))
    finish(ax, [stepped(ax, val, 'load_bearing', blue, 'load-bearing'), stepped(ax, val, 'support', orange, 'support')])

    ax = axes[1, 1]
    style(ax, 'Prompts with a learning signal', 'share of training prompts whose samples differ in reward', (0, 1.04))
    finish(ax, [noisy(ax, train, 'informative_prompts', blue, None)], legend=False)

    ax = axes[1, 2]
    style(ax, 'Seconds per step', 'wall-clock seconds, one GPU step')
    ends = [noisy(ax, train, 'sec_per_step', blue, 'whole step'), noisy(ax, train, 'sample_sec', orange, 'sampling')]
    ax.set_ylim(bottom=0)
    finish(ax, ends, fmt='{:.1f}')

    ax = axes[1, 3]
    style(ax, 'Peak GPU memory', 'GB, maximum over the GPUs')
    ends = [noisy(ax, train, 'mem_train_gb', blue, 'gradient step'), noisy(ax, train, 'mem_sample_gb', orange, 'sampling')]
    ax.set_ylim(bottom=0)
    finish(ax, ends, fmt='{:.0f}')

    for ax in axes[1]:
        ax.set_xlabel('training step', fontsize=8.5)
    last = train[-1]
    head = f'{run_dir.name}: verifier-guided RL'
    sub = f'step {last["step"]:,} of {last.get("of", last["step"]):,}'
    if last.get('hours') is not None:
        sub += f' · {last["hours"]:.1f} h'
    if val:
        sub += f' · validation accuracy {val[0].get("correct", float("nan")):.3f} → {val[-1].get("correct", float("nan")):.3f}' if len(val) > 1 else f' · validation accuracy {val[-1].get("correct", float("nan")):.3f}'
    fig.text(0.045, 0.962, head, fontsize=14, fontweight='bold', color=INK, va='center')
    fig.text(0.045, 0.928, sub, fontsize=9.5, color=INK_2, va='center')
    out = Path(out) if out else run_dir / 'curves.png'
    fig.savefig(out, dpi=170)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
