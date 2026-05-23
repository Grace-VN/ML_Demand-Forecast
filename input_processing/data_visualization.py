# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import matplotlib 
# Use Agg backend for script execution in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT_DIR = Path(__file__).parent.parent


def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
# If ensure_import_path() modifies sys.path, run it first:
ensure_import_path() 

# Now import df globally so it's available everywhere in this file
from input_processing.data_loading import df

def plot_demand_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['Demand'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_title('Demand Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Demand (Units)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df['Demand'].mean(), color='red', linestyle='--', label=f'Mean: {df["Demand"].mean():.1f}')
    axes[0].legend()

    df.boxplot(column='Demand', by='Category', ax=axes[1], patch_artist=True, figsize=(14, 5))
    axes[1].set_title('Demand by Product Category', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Demand')
    plt.suptitle('')

    plt.tight_layout()
    output_path_0 = ROOT_DIR / 'chart_demand_distribution.png'
    plt.savefig(output_path_0, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path_0


def print_summary(df, chart_path):
    print(f"Saved chart: {chart_path}")
    print(f"\nAverage Demand: {df['Demand'].mean():.1f} units")
    print(f"Max Demand:     {df['Demand'].max()} units")
    print(f"Min Demand:     {df['Demand'].min()} units")

# ── Chart 2: Demand by Region & Promotion ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Average demand per region
region_demand = df.groupby('Region')['Demand'].mean().sort_values(ascending=False)
bars = axes[0].bar(region_demand.index, region_demand.values, 
                   color=['#2196F3','#4CAF50','#FF9800','#E91E63'], edgecolor='white')
axes[0].set_title(' Avg Demand by Region', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Average Demand')
# Add value labels on bars
for bar, val in zip(bars, region_demand.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}', ha='center', fontweight='bold')

# Promotion vs No Promotion
promo_demand = df.groupby('Promotion')['Demand'].mean()
labels = ['No Promotion', 'With Promotion']
colors = ['#90CAF9', '#1565C0']
axes[1].bar(labels, promo_demand.values, color=colors, edgecolor='white')
axes[1].set_title(' Promotion Impact on Demand', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Average Demand')
for i, val in enumerate(promo_demand.values):
    axes[1].text(i, val + 1, f'{val:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
output_path_1 = ROOT_DIR / 'output_storage' / 'images' / 'chart_region_promo.png'
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)

# ── Chart 3: Correlation Heatmap ──────────────────────────────────────────────
# A heatmap shows how strongly each column relates to another
# (Closer to 1 or -1 = stronger relationship)

numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 
                'Price', 'Discount', 'Competitor Pricing', 'Demand']

plt.figure(figsize=(10, 7))
corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix, 
            annot=True,          # Show numbers in each cell
            fmt='.2f',           # Round to 2 decimal places
            cmap='coolwarm',     # Color scheme: blue=negative, red=positive
            center=0,
            square=True,
            linewidths=0.5)

plt.title(' Correlation Heatmap\n(How closely each column is related)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
output_path_2 = ROOT_DIR / 'output_storage' / 'images' / 'chart_heatmap.png'
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')
plt.show()

# Print top correlations with Demand
print("\n🎯 How each column correlates with DEMAND:")
print(corr_matrix['Demand'].sort_values(ascending=False).to_string())
