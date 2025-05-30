import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def plot_family_recall(metrics_csv: str,
                       output_path: str = None,
                       error_type: str = 'std'):
    """
    Plots mean recall by model family with error bars indicating variability across models.
    Improvements:
      - Uses an explicit Figure/Axes for finer control.
      - Adds minor ticks and a two-tier grid.
      - Increases font sizes for publication clarity.
      - Tightens legend placement and styling.
    """
    base = os.path.basename(metrics_csv)
    dataset_name = os.path.splitext(base)[0].replace('_metrics', '')
    # --- Load & prepare data ---
    df = pd.read_csv(metrics_csv)
    def get_family(m: str) -> str:
        m = m.lower()
        for prefix, fam in [
            ('gpt',   'GPT'),
            ('claude','Claude'),
            ('gemini','Gemini'),
            ('phi',   'Phi'),
            ('llama', 'LLaMA'),
            ('idefics','Idefics'),
            ('paligemma','Paligemma'),
            ('qwen',  'Qwen'),
        ]:
            if m.startswith(prefix):
                return fam
        return 'Other'
    df['Family'] = df['Model'].apply(get_family)

    distances = [1, 5, 10, 20, 100, 200, 750]
    recall_cols = [f'R@{d}km' for d in distances]
    df[recall_cols] = df[recall_cols].apply(pd.to_numeric)

    fam_stats = df.groupby('Family')[recall_cols]
    fam_mean = fam_stats.mean()
    fam_err  = fam_stats.std() if error_type=='std' else fam_stats.sem()

    fig, ax = plt.subplots(figsize=(10,6))
    for fam in fam_mean.index:
        x = distances
        y = fam_mean.loc[fam].values
        e = fam_err.loc[fam].values

        ax.plot(x, y, marker='o', linewidth=2, markersize=6, label=fam)
        ax.fill_between(x, y-e, y+e, alpha=0.25)

    ax.set_xscale('log')
    ax.set_xticks(distances)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', alpha=0.5)
    ax.grid(which='minor', linestyle=':',  alpha=0.3)

    ax.set_xlabel('Distance Threshold (km)', fontsize=13)
    ax.set_ylabel('Mean Recall (%)',         fontsize=13)
    ax.set_title(f'Geolocation Accuracy by Family on {dataset_name}',
                  fontsize=14, fontweight='bold')

    ax.legend(title='Family', frameon=False,
              loc='upper left', bbox_to_anchor=(1.02,1))
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot geolocation recall metrics with enhanced styling.'
    )
    parser.add_argument('--csv',   type=str,
                        default='results/metrics/im2gps_metrics.csv')
    parser.add_argument('--output',type=str, default=None)
    parser.add_argument('--error', type=str, choices=['std','sem'],
                        default='std')
    args = parser.parse_args()
    plot_family_recall(args.csv, args.output, args.error)
