import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os 


def plot_recall(metrics_csv: str, output_path: str = None):
    """
    Load a CSV with geolocation recall metrics and plot recall vs. distance for each model with enhanced styling.
    """
    base = os.path.basename(metrics_csv)
    dataset_name = os.path.splitext(base)[0].replace('_metrics', '')

    # Read the metrics table
    df = pd.read_csv(metrics_csv)

    # Define the distance thresholds and corresponding column names
    distances = [1, 5, 10, 20, 100, 200, 750]
    recall_cols = [f'R@{d}km' for d in distances]

    # Ensure recall columns are numeric
    df[recall_cols] = df[recall_cols].apply(pd.to_numeric)

    # Apply a clean, modern style
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose a qualitative colormap for distinct lines
    cmap = plt.get_cmap('tab10')

    # Plot each model's recall curve
    for idx, row in df.iterrows():
        model = row['Model']
        recalls = row[recall_cols].values
        ax.plot(
            distances,
            recalls,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=6,
            label=model,
            color=cmap(idx % cmap.N)
        )

    # Log scale for x-axis and custom ticks
    ax.set_xscale('log')
    ax.set_xticks(distances)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Labels and title with improved font sizing
    ax.set_xlabel('Distance Threshold (km)', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title(f'Geolocation Accuracy on {dataset_name}',
                  fontsize=14, fontweight='bold')

    # Grid, legend, and layout adjustments
    ax.grid(which='both', linestyle='--', alpha=0.6, linewidth=0.7)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize='small')
    plt.tight_layout()

    # Save or show
    if output_path:
        fig.savefig(output_path, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot geolocation recall metrics from a CSV file with enhanced styling.'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='results/metrics/im2gps_metrics.csv',
        help='Path to the metrics CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots/images/fig_improved.png',
        help='Optional path to save the enhanced figure'
    )
    args = parser.parse_args()
    plot_recall(args.csv, args.output)
