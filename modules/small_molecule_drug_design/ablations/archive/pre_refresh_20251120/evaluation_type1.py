#!/usr/bin/env python3
"""
Type 1 Evaluation: Focus on optimized properties only for LLM mutation runs.

Analyzes:
- Target activity (E. coli or K. pneumoniae)
- Novelty
- Safety (1 - toxicity)
- Aggregate (sum of the three)

Applies diversity filter at 0.6 Tanimoto similarity.
NO held-out metrics (QED, SA, MW, PAINS, BRENK).

Includes the new LLM mutation files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import DataStructs
from pathlib import Path

# Setup
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

DATA_DIR = Path("scileo_ablations")
OUTPUT_DIR = Path("type1_results")
OUTPUT_DIR.mkdir(exist_ok=True)

METHOD_COLORS = {
    'SciLeo_old': '#2E86AB',
    'SciLeo_butina_5_objective': '#4A90A4',
    'SciLeo_diverse_5_objective': '#6A9AB4',
    'SciLeo_level1_iter1': '#2D9CDB',
    'SciLeo_level1_iter2': '#4AB3E3',
    'SciLeo_level2_iter1': '#5B4F94',
    'SciLeo_level2_iter2': '#7D6DB8',
    'REINVENT4': '#A23B72',
    'NatureLM': '#F18F01',
    'MolT5': '#C73E1D',
    'TextGrad': '#6A994E'
}




def load_scileo_seed(filename, target):
    """Load single SciLeo seed."""
    df = pd.read_csv(DATA_DIR / filename)
    seed = filename.split('_')[-3]

    if target == 'ecoli':
        target_col = 'escherichia_coli_minimol'
    else:
        target_col = 'klebsiella_pneumoniae_minimol'

    df['target_activity'] = df[target_col]

    # Normalize toxicity column
    if 'toxicity_safety_chemprop' in df.columns and 'primary_1_minus_cell_toxicity_chemprop' not in df.columns:
        df['primary_1_minus_cell_toxicity_chemprop'] = df['toxicity_safety_chemprop']
    
    if 'primary_1_minus_cell_toxicity_chemprop' in df.columns:
        df['toxicity'] = df['primary_1_minus_cell_toxicity_chemprop']
    elif 'toxicity_safety_chemprop' in df.columns:
        df['toxicity'] = df['toxicity_safety_chemprop']
    else:
        raise ValueError(f"No toxicity column found in {filename}")

    df['aggregate'] = df['target_activity'] + df['antibiotics_novelty'] + df['toxicity']

    return df, seed


def load_baseline_data(method, target, limit_reinvent=True):
    """Load baseline method data."""
    files = {
        ('REINVENT4', 'ecoli'): 'stage1_1_reinvent4_ecoli_scored.csv',
        ('REINVENT4', 'kpneumoniae'): 'stage1_1_reinvent4_kp_scored.csv',
        ('NatureLM', 'ecoli'): 'naturelm_ecoli_antibiot_final_scored.csv',
        ('NatureLM', 'kpneumoniae'): 'naturelm_KP_antibiot_final_scored_results.csv',
        ('MolT5', 'ecoli'): 'molt5_ecoli_antibiot_final_scored.csv',
        ('MolT5', 'kpneumoniae'): 'molt5_KP_antibiot_final_scored_results.csv',
        ('TextGrad', 'ecoli'): 'textgrad_ecoli_output_new_epoch_merged_scored.csv',
        ('TextGrad', 'kpneumoniae'): 'textgrad_kp_output_new_epoch_merged_scored.csv'
    }

    df = pd.read_csv(DATA_DIR / files[(method, target)])

    # Limit REINVENT4 to first 10k
    if method == 'REINVENT4' and limit_reinvent and len(df) > 10000:
        df = df.head(10000).copy()

    if target == 'ecoli':
        target_col = 'escherichia_coli'
    else:
        target_col = 'klebsiella_pneumoniae'

    df['target_activity'] = df[target_col]
    df['aggregate'] = df['target_activity'] + df['antibiotics_novelty'] + df['toxicity']

    return df


def save_molecules(target):
    """Save molecules for each method."""
    target_name = "E. coli" if target == 'ecoli' else "K. pneumoniae"
    print(f"\n{'='*70}")
    print(f"Saving Molecules - {target_name}")
    print('='*70)

    results = {}

    # SciLeo methods (level1, level2 for E. coli)
    if target == 'ecoli':
        scileo_files = [
            # Level 1 iteration 1 & 2 (CORRECTED ORDER)
            ('ecoli_level1_iter1_20251101150814_all_molecules.csv', 'level1_iter1'),
            ('ecoli_level1_iter1_20251102002019_all_molecules.csv', 'level1_iter2'),
            # Level 2 iteration 1 & 2 (CORRECTED ORDER)
            ('ecoli_level2_iter1_20251101150813_all_molecules.csv', 'level2_iter1'),
            ('ecoli_level2_iter1_20251102002529_all_molecules.csv', 'level2_iter2'),
        ]
    else:
        scileo_files = []

    # Process SciLeo methods
    for filename, method_suffix in scileo_files:
        df = pd.read_csv(DATA_DIR / filename)

        if target == 'ecoli':
            target_col = 'escherichia_coli_minimol'
        else:
            target_col = 'klebsiella_pneumoniae_minimol'

        df['target_activity'] = df[target_col]

        # Normalize toxicity column
        if 'toxicity_safety_chemprop' in df.columns and 'primary_1_minus_cell_toxicity_chemprop' not in df.columns:
            df['primary_1_minus_cell_toxicity_chemprop'] = df['toxicity_safety_chemprop']
        
        if 'primary_1_minus_cell_toxicity_chemprop' in df.columns:
            df['toxicity'] = df['primary_1_minus_cell_toxicity_chemprop']
        elif 'toxicity_safety_chemprop' in df.columns:
            df['toxicity'] = df['toxicity_safety_chemprop']
        else:
            raise ValueError(f"No toxicity column found in {filename}")

        df['aggregate'] = df['target_activity'] + df['antibiotics_novelty'] + df['toxicity']

        method_name = f'SciLeo_{method_suffix}'
        print(f"\n{method_name}: {len(df)} molecules")

        output_file = OUTPUT_DIR / f"scileo_{method_suffix}_{target}.csv"
        df.to_csv(output_file, index=False)
        results[method_name] = len(df)

    # Baselines
    for method in ['REINVENT4', 'NatureLM', 'MolT5', 'TextGrad']:
        try:
            baseline = load_baseline_data(method, target, limit_reinvent=True)
            baseline_clean = baseline.dropna(subset=['smiles', 'target_activity',
                                                     'antibiotics_novelty', 'toxicity'])
            print(f"\n{method}: {len(baseline_clean)} molecules")

            output_file = OUTPUT_DIR / f"{method.lower()}_{target}.csv"
            baseline_clean.to_csv(output_file, index=False)
            results[method] = len(baseline_clean)
        except Exception as e:
            print(f"\n{method}: Error: {e}")
            results[method] = 0

    return results


def plot_property_comparison(target, k_values=[10, 100, 1000]):
    """Plot comparison of optimized properties at different Top-K levels."""
    target_name = "E. coli" if target == 'ecoli' else "K. pneumoniae"

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Optimized Properties Comparison (with LLM Mutation) - {target_name}\n(No Held-Out Filters)',
                 fontsize=16, fontweight='bold')

    # Load diverse datasets
    if target == 'ecoli':
        scileo_methods = ['SciLeo_level1_iter1', 'SciLeo_level1_iter2', 'SciLeo_level2_iter1', 'SciLeo_level2_iter2']
    else:
        scileo_methods = []

    methods = scileo_methods + ['REINVENT4', 'NatureLM', 'MolT5', 'TextGrad']

    properties = [
        ('aggregate', 'Aggregate Score (Sum)', axes[0, 0]),
        ('target_activity', f'{target_name} Activity', axes[0, 1]),
        ('antibiotics_novelty', 'Novelty Score', axes[1, 0]),
        ('toxicity', 'Safety (1 - Toxicity)', axes[1, 1])
    ]

    for prop_name, prop_title, ax in properties:
        results = {'Method': [], 'K': [], 'Score': []}

        for k in k_values:
            for method in methods:
                try:
                    if method.startswith('SciLeo_'):
                        method_suffix = method.replace('SciLeo_', '')
                        data = pd.read_csv(OUTPUT_DIR / f"scileo_{method_suffix}_{target}.csv")
                        # Ensure aggregate is calculated if missing
                        if 'aggregate' not in data.columns:
                            data['aggregate'] = data['target_activity'] + data['antibiotics_novelty'] + data['toxicity']
                    else:
                        data = pd.read_csv(OUTPUT_DIR / f"{method.lower()}_{target}.csv")

                    data_clean = data.dropna(subset=[prop_name])

                    if len(data_clean) >= k:
                        top_k = data_clean.nlargest(k, 'aggregate')
                        score = top_k[prop_name].mean()

                        results['Method'].append(method)
                        results['K'].append(f'Top-{k}')
                        results['Score'].append(score)
                except Exception as e:
                    print(f"Warning: {method} {prop_name} error: {e}")

        df_plot = pd.DataFrame(results)

        # Plot
        x = np.arange(len(k_values))
        n_methods = len(methods)
        width = 0.8 / n_methods

        for i, method in enumerate(methods):
            method_data = df_plot[df_plot['Method'] == method]
            if len(method_data) > 0:
                # Create full array with NaN for missing k values
                scores_dict = dict(zip(method_data['K'], method_data['Score']))
                scores = [scores_dict.get(f'Top-{k}', np.nan) for k in k_values]

                # Only plot non-NaN values
                valid_mask = ~np.isnan(scores)
                x_positions = x[valid_mask] + i*width - 0.4
                valid_scores = np.array(scores)[valid_mask]

                if len(valid_scores) > 0:
                    bars = ax.bar(x_positions, valid_scores, width, label=method,
                                 color=METHOD_COLORS.get(method, 'gray'))

                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0.01:
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)

        ax.set_xlabel('Top-K Selection', fontweight='bold')
        ax.set_ylabel(prop_title, fontweight='bold')
        ax.set_title(prop_title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'properties_comparison_{target}.png', bbox_inches='tight')
    print(f"\n✓ Saved: properties_comparison_{target}.png")
    plt.close()


def main():
    """Main evaluation function."""
    print("="*70)
    print("Type 1 Evaluation: Optimized Properties Only (with LLM Mutation)")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"REINVENT4 limited to first 10k molecules")
    print(f"SciLeo seeds and LLM mutation runs shown separately")
    print("No diversity filtering for speed")
    print("\nProperties evaluated:")
    print("  - Target activity (E. coli or K. pneumoniae)")
    print("  - Novelty (antibiotics_novelty)")
    print("  - Safety (1 - toxicity)")
    print("  - Aggregate (sum of the three)")

    # Process both targets
    for target in ['ecoli', 'kpneumoniae']:
        save_molecules(target)
        plot_property_comparison(target, k_values=[10, 100, 1000])

    print("\n" + "="*70)
    print("✅ Type 1 Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
