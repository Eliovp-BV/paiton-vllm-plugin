"""Render the APC report figure from exported measurements (Matplotlib 3.11.2)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path(__file__).resolve().parents[1]
    data = json.loads((base / 'assets/plot-data.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.hashsalt': 'paiton-apc-20260917'})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.subplots_adjust(left=.065, right=.985, top=.72, bottom=.27, wspace=.22)
    fig.suptitle('Prefix reuse removes most repeat-turn waiting', x=.065, y=.97,
                 ha='left', fontsize=22, fontweight='bold', color='#0f172a')
    fig.text(.065, .91, 'Qwen3.8 27B MXFP4 + DFlash2  |  One Radeon AI PRO R9700  |  17 September 2026',
             fontsize=12, color='#475569')
    colors = {'a': '#64748b', 'c': '#3b82f6', 'd': '#16a34a'}
    labels = {'a': 'Released compact native / APC off',
              'c': 'Stock GDN / APC on',
              'd': 'Native-prefill APC candidate'}
    x = np.arange(4)
    for index, arm in enumerate(('a', 'c', 'd')):
        values = [r['ttft_seconds'] for r in data['forty_k'][arm]['cases']]
        bars = axes[0].bar(x + (index - 1) * .26, values, width=.24,
                           color=colors[arm], label=labels[arm], zorder=3)
        axes[0].bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=9)
    axes[0].set_xticks(x, ['Cold', 'Repeat', 'Grow 1', 'Grow 2'])
    axes[0].set_ylim(0, 18)
    axes[0].set_title('40K prompts: matched 5 GiB cache', loc='left', fontsize=14, pad=16)
    axes[0].set_ylabel('First-token latency (seconds) — lower is better')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left',
               bbox_to_anchor=(.06, .86), ncol=3, frameon=False, fontsize=10)

    rows = data['long_context']['primary']['cases']
    values = [r['ttft_seconds'] for r in rows]
    bars = axes[1].bar(x, values, width=.62, color=['#64748b', '#16a34a', '#16a34a', '#16a34a'], zorder=3)
    axes[1].bar_label(bars, labels=[f'{v:.2f} s' for v in values], padding=5, fontsize=11)
    axes[1].set_xticks(x, ['Cold', 'Repeat', 'Grow 1', 'Grow 2'])
    axes[1].set_ylim(0, 105)
    axes[1].set_title('150K prompts: same APC candidate, 8 GiB cache', loc='left', fontsize=14, pad=16)
    axes[1].set_ylabel('First-token latency (seconds)')
    axes[1].text(.96, .86, f'{values[0]/values[1]:.2f}×', transform=axes[1].transAxes,
                 ha='right', fontsize=30, fontweight='bold', color='#15803d')
    axes[1].text(.96, .75, 'cold / repeated-prefix latency', transform=axes[1].transAxes,
                 ha='right', fontsize=11, color='#475569')
    confirm = data['long_context']['confirmation']['cases']
    axes[1].text(0, -.22, f"Independent confirmation: {confirm[0]['ttft_seconds']:.2f} s → "
                 f"{confirm[1]['ttft_seconds']:.2f} s", transform=axes[1].transAxes,
                 fontsize=11, color='#334155')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#cbd5e1')
        ax.tick_params(colors='#334155')
        ax.grid(axis='y', color='#e2e8f0', linewidth=.8, zorder=0)
    fig.text(.065, .055, 'One active request; client TTFT includes queueing. Cold → repeat → two growing turns uses fixed answer history.\n'
             '150K uses an unreleased native-prefill APC candidate and a 160K context limit. Decode and capacity tradeoffs are in the report.',
             fontsize=10, color='#475569', linespacing=1.6)
    for extension in ('png', 'svg'):
        metadata = {'Software': 'Matplotlib'} if extension == 'png' else {'Date': None, 'Creator': 'Matplotlib'}
        fig.savefig(base / f'assets/prefix-reuse-latency.{extension}', dpi=160,
                    facecolor='white', metadata=metadata)
        if extension == 'svg':
            path = base / 'assets/prefix-reuse-latency.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
