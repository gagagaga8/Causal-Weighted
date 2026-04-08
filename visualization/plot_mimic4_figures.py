"""
Publication-quality visualization - MIMIC-IV version
Generate figures directly from ablation experiment results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

os.chdir('c:/Dynamic-RRT')

# Set plot style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")
sns.set_palette("Set2")

OUTPUT_DIR = Path("Visualization/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_baseline_comparison():
    """Fig1: BaselineMethodComparison"""
    # BaselineModelPerformanceData ExperimentResults 
    data = {
        'model': ['Logistic Regression', 'Random Forest', 'XGBoost', 
                  'Gradient Boosting', 'MLP (128-64)', 'LightGBM (Ours)'],
        'adr': [0.7655, 0.8956, 0.9077, 0.9317, 0.9224, 0.9301],
        'std': [0.0189, 0.0095, 0.0105, 0.0033, 0.0063, 0.0069]
    }
    df = pd.DataFrame(data)
    df = df.sort_values('adr', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#E74C3C' if 'Ours' in x else '#3498DB' for x in df['model']]
    bars = ax.barh(df['model'], df['adr'], xerr=df['std'], 
                   color=colors, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Accurate Decision Rate (ADR)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Method Comparison on MIMIC-IV', fontsize=14, fontweight='bold')
    ax.set_xlim(0.7, 1.0)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Threshold')
    
    for i, (model, adr) in enumerate(zip(df['model'], df['adr'])):
        ax.text(adr + 0.01, i, f'{adr:.2%}', va='center', fontsize=9)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[1/8] baseline_comparison.png ✓")


def plot_timepoint_ablation():
    """Fig2: time pointAblationExperiment"""
    try:
        df = pd.read_csv('Experiment /ablation_timepoint.csv')
    except:
        # Use experimental results data
        data = {
            'config': ['Full Model', 'Baseline Only', 'Baseline + k1', 
                      'Baseline + k1 + k2', 'w/o k1', 'w/o k2', 'w/o k3'],
            'n_features': [26, 11, 16, 21, 21, 21, 21],
            'adr': [0.9301, 0.9038, 0.9132, 0.9269, 0.9295, 0.9225, 0.9269],
            'drop': [0, 0.0263, 0.0169, 0.0032, 0.0006, 0.0076, 0.0032]
        }
        df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature
    ax = axes[0]
    cumulative = df[df['config'].isin(['Baseline Only', 'Baseline + k1', 
                                        'Baseline + k1 + k2', 'Full Model'])]
    cumulative = cumulative.sort_values('n_features')
    
    ax.plot(cumulative['n_features'], cumulative['adr'], 
            marker='o', linewidth=2, markersize=10, color='#3498DB')
    ax.fill_between(cumulative['n_features'], cumulative['adr'], 
                    alpha=0.2, color='#3498DB')
    
    for i, row in cumulative.iterrows():
        ax.annotate(row['config'], (row['n_features'], row['adr']),
                   textcoords="offset points", xytext=(5, 10), fontsize=9)
    
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('ADR', fontsize=12, fontweight='bold')
    ax.set_title('Feature Accumulation Effect', fontsize=12, fontweight='bold')
    ax.set_ylim(0.89, 0.94)
    ax.grid(True, alpha=0.3)
    
    # time point
    ax = axes[1]
    removal = df[df['config'].str.startswith('w/o')]
    
    colors = ['#E74C3C' if row['drop'] == removal['drop'].max() else '#F39C12' 
              for _, row in removal.iterrows()]
    bars = ax.bar(removal['config'], removal['drop'] * 100, color=colors, alpha=0.8)
    
    ax.set_xlabel('Removed Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Drop (%)', fontsize=12, fontweight='bold')
    ax.set_title('Timepoint Contribution (Removal Impact)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, drop in zip(bars, removal['drop']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{drop*100:.2f}%', ha='center', fontsize=10)
    
    plt.suptitle('Temporal Feature Ablation Study (MIMIC-IV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timepoint_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[2/8] timepoint_ablation.png ✓")


def plot_hyperparameter_ablation():
    """Fig3: Parameters Analysis"""
    try:
        df = pd.read_csv('Experiment /ablation_hyperparams.csv')
    except:
        data = {
            'config': ['Default (500,10,0.05)', 'n_est=100', 'n_est=1000',
                      'depth=5', 'depth=15', 'lr=0.01', 'lr=0.1'],
            'adr': [0.9301, 0.8904, 0.9343, 0.9209, 0.9330, 0.8875, 0.9338]
        }
        df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_sorted = df.sort_values('adr', ascending=True)
    colors = ['#2ECC71' if 'Default' in str(x) else '#3498DB' for x in df_sorted['config']]
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['adr'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['config'])
    ax.set_xlabel('ADR', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Sensitivity Analysis (MIMIC-IV)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.85, 0.95)
    ax.axvline(x=0.93, color='red', linestyle='--', alpha=0.5)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['adr'] + 0.002, i, f'{row["adr"]:.2%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[3/8] hyperparameter_sensitivity.png ✓")


def plot_data_efficiency():
    """Fig4: Data Curve"""
    try:
        df = pd.read_csv('Experiment /ablation_data_ratio.csv')
    except:
        data = {
            'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'n_train': [380, 760, 1141, 1521, 1902, 2282, 2662, 3043, 3423],
            'adr': [0.9216, 0.9246, 0.9282, 0.9289, 0.9264, 0.9280, 0.9271, 0.9311, 0.9286],
            'std': [0.0029, 0.0049, 0.0021, 0.0040, 0.0034, 0.0049, 0.0066, 0.0050, 0.0113]
        }
        df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(df['ratio']*100, df['adr'], yerr=df['std'], 
                marker='o', linewidth=2, markersize=8, capsize=5,
                color='#3498DB', label='ADR ± Std')
    ax.fill_between(df['ratio']*100, df['adr']-df['std'], df['adr']+df['std'],
                    alpha=0.2, color='#3498DB')
    
    ax.axhline(y=0.92, color='red', linestyle='--', alpha=0.5, label='92% Threshold')
    ax.set_xlabel('Training Data Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ADR', fontsize=12, fontweight='bold')
    ax.set_title('Data Efficiency: Learning Curve (MIMIC-IV)', fontsize=14, fontweight='bold')
    ax.set_ylim(0.90, 0.95)
    ax.set_xlim(5, 95)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.annotate('10% data achieves 92%+', xy=(10, 0.9216), 
               xytext=(25, 0.915), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[4/8] data_efficiency.png ✓")


def plot_class_weight_analysis():
    """Fig5: ClassWeightAnalysis"""
    try:
        df = pd.read_csv('Experiment /ablation_class_weight.csv')
    except:
        data = {
            'config': ['None', 'balanced', 'RRT×5', 'RRT×10', 'RRT×15', 'RRT×20', 'RRT×30'],
            'adr': [0.9374, 0.9301, 0.9335, 0.9290, 0.9309, 0.9274, 0.9277],
            'treat_acc': [0.1497, 0.2147, 0.1863, 0.1943, 0.2267, 0.2309, 0.2431],
            'control_acc': [0.9921, 0.9798, 0.9854, 0.9800, 0.9798, 0.9758, 0.9753]
        }
        df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ADR vs Weight
    ax = axes[0]
    x = range(len(df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df['adr'], width, 
                   label='Overall ADR', color='#3498DB', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], df['treat_acc'], width,
                   label='RRT Group Acc', color='#E74C3C', alpha=0.8)
    
    ax.set_xlabel('Class Weight Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Class Weight Impact', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['config'], rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Curve
    ax = axes[1]
    ax.scatter(df['adr'], df['treat_acc'], s=100, c=range(len(df)), 
               cmap='coolwarm', alpha=0.8, edgecolors='black')
    
    for i, row in df.iterrows():
        ax.annotate(row['config'], (row['adr'], row['treat_acc']),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('Overall ADR', fontsize=12, fontweight='bold')
    ax.set_ylabel('RRT Group Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('ADR vs RRT Sensitivity Trade-off', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    best_idx = df['treat_acc'].idxmax()
    ax.scatter(df.loc[best_idx, 'adr'], df.loc[best_idx, 'treat_acc'], 
              s=200, marker='*', color='gold', edgecolors='black', zorder=5)
    
    plt.suptitle('Class Weight Optimization (MIMIC-IV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_weight_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[5/8] class_weight_analysis.png ✓")


def plot_feature_type_ablation():
    """Fig6: FeaturetypeAblation"""
    try:
        df = pd.read_csv('Experiment /ablation_feature_type.csv')
    except:
        data = {
            'feature_type': ['Full Model', 'Demographic', 'Severity', 'Renal', 
                            'Urine', 'Metabolic', 'Acid-Base'],
            'n_features': [26, 3, 4, 4, 6, 6, 3],
            'adr': [0.9322, 0.7947, 0.5560, 0.8730, 0.7379, 0.9143, 0.7826]
        }
        df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # useeach type
    ax = axes[0]
    single = df[df['feature_type'] != 'Full Model']
    single_sorted = single.sort_values('adr', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(single_sorted)))
    bars = ax.barh(single_sorted['feature_type'], single_sorted['adr'], 
                   color=colors, alpha=0.8)
    
    ax.set_xlabel('ADR (Standalone)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Type Performance (Single Category)', fontsize=12, fontweight='bold')
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5, label='90% Target')
    ax.set_xlim(0.5, 1.0)
    
    for bar, adr in zip(bars, single_sorted['adr']):
        ax.text(adr + 0.01, bar.get_y() + bar.get_height()/2,
               f'{adr:.2%}', va='center', fontsize=9)
    
    # Features vsPerformance
    ax = axes[1]
    ax.scatter(single['n_features'], single['adr'], s=150, 
               c=range(len(single)), cmap='Set2', alpha=0.8, edgecolors='black')
    
    for i, row in single.iterrows():
        ax.annotate(row['feature_type'], (row['n_features'], row['adr']),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('ADR', fontsize=12, fontweight='bold')
    ax.set_title('Feature Count vs Performance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Type Ablation Study (MIMIC-IV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_type_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[6/8] feature_type_ablation.png ✓")


def plot_external_validation():
    """Fig7: ExternalValidationResults"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Internal vsExternalValidation
    ax = axes[0]
    datasets = ['MIMIC-IV\n(Internal)', 'eICU\n(External)']
    auc_values = [0.896, 0.720]
    adr_values = [0.9301, 0.6928]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auc_values, width, label='AUC-ROC', 
                   color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, adr_values, width, label='ADR',
                   color='#E74C3C', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Internal vs External Validation', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', fontsize=10)
    
    ax = axes[1]
    methods = ['Direct\nTransfer', 'Feature\nSelection', 'Platt\nScaling', 
               'Under-\nsampling', 'Ensemble']
    auc_da = [0.696, 0.695, 0.704, 0.720, 0.697]
    rrt_sens = [0.4462, 0.4117, 0.6478, 0.8118, 0.4934]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auc_da, width, label='AUC', color='#9B59B6', alpha=0.8)
    bars2 = ax.bar(x + width/2, rrt_sens, width, label='RRT Sensitivity', 
                   color='#F39C12', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Domain Adaptation Strategies on eICU', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('External Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'external_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[7/8] external_validation.png ✓")


def plot_summary_figure():
    """Fig8: Summary - All Results"""
    fig = plt.figure(figsize=(16, 12))
    
    # 2x2 
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. toPerformanceMetric
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Overall ADR', 'RRT Group Acc', 'Non-RRT Acc', 'AUC-ROC']
    values = [0.9301, 0.2147, 0.9798, 0.896]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('A. Main Performance Metrics (MIMIC-IV)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2%}', 
                ha='center', fontsize=10)
    
    # 2. time point 
    ax2 = fig.add_subplot(gs[0, 1])
    timepoints = ['k1\n(6-12h)', 'k2\n(12-24h)', 'k3\n(24-48h)']
    contributions = [0.06, 0.76, 0.32]
    colors = ['#F39C12' if c == max(contributions) else '#3498DB' for c in contributions]
    
    bars = ax2.bar(timepoints, contributions, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Performance Drop (%)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Timepoint Contribution (Removal Impact)', fontsize=12, fontweight='bold')
    for bar, v in zip(bars, contributions):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}%', 
                ha='center', fontsize=10)
    ax2.annotate('k2 most critical', xy=(1, 0.76), xytext=(1.5, 0.6),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    # 3. FeaturetypePerformance
    ax3 = fig.add_subplot(gs[1, 0])
    types = ['Metabolic', 'Renal', 'Acid-Base', 'Demographic', 'Urine', 'Severity']
    standalone_adr = [0.9143, 0.8730, 0.7826, 0.7947, 0.7379, 0.5560]
    
    bars = ax3.barh(types, standalone_adr, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(types))), 
                    alpha=0.8, edgecolor='black')
    ax3.set_xlabel('ADR (Standalone)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Feature Type Importance', fontsize=12, fontweight='bold')
    ax3.axvline(x=0.9, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlim(0.5, 1.0)
    for bar, v in zip(bars, standalone_adr):
        ax3.text(v + 0.01, bar.get_y() + bar.get_height()/2, f'{v:.1%}', 
                va='center', fontsize=9)
    
    # 4. ExternalValidationComparison
    ax4 = fig.add_subplot(gs[1, 1])
    categories = ['MIMIC-IV\n(Development)', 'eICU\n(Direct)', 'eICU\n(Adapted)']
    auc_values = [0.896, 0.696, 0.720]
    
    bars = ax4.bar(categories, auc_values, color=['#2ECC71', '#E74C3C', '#3498DB'], 
                   alpha=0.8, edgecolor='black')
    ax4.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax4.set_title('D. Multi-center Validation', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    for bar, v in zip(bars, auc_values):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', 
                ha='center', fontsize=10)
    
    plt.suptitle('Dynamic RRT Decision Support System - Key Results Summary\n(Development: MIMIC-IV | External Validation: eICU)', 
                fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'summary_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[8/8] summary_figure.png ✓")


def generate_all_figures():
    """GenerateAll Figure"""
    print("="*60)
    print("StartGeneratepaperFigure (MIMIC-IVVersion)")
    print("="*60)
    
    plot_baseline_comparison()
    plot_timepoint_ablation()
    plot_hyperparameter_ablation()
    plot_data_efficiency()
    plot_class_weight_analysis()
    plot_feature_type_ablation()
    plot_external_validation()
    plot_summary_figure()
    
    print("="*60)
    print(f"All FigurealreadySave : {OUTPUT_DIR}")
    print("="*60)
    
    # Column GenerateFile
    print("\nGenerateFigureFile:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    generate_all_figures()
