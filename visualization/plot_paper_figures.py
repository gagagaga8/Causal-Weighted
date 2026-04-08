"""
Publication-quality visualization
Generate all experiment result figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")
sns.set_palette("Set2")

OUTPUT_DIR = Path("Visualization/figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_baseline_comparison():
    """BaselineMethodComparison"""
    df = pd.read_csv('Experiment/baseline_comparison.csv')
    df = df.sort_values('adr', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#E74C3C' if 'Ours' in x else '#3498DB' for x in df['model']]
    bars = ax.barh(df['model'], df['adr'], xerr=df['std'], 
                   color=colors, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0.7, 1.0)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Threshold')
    
    # valueLabel
    for i, (model, adr) in enumerate(zip(df['model'], df['adr'])):
        ax.text(adr + 0.01, i, f'{adr:.2%}', va='center', fontsize=9)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("alreadyGenerate: baseline_comparison.png")


def plot_ablation_study():
    """AblationExperimentFeature to """
    df = pd.read_csv('Experiment/ablation_results.csv')
    df_wo = df[df['feature_set'].str.startswith('w/o')].copy()
    df_wo['feature'] = df_wo['feature_set'].str.replace('w/o ', '')
    df_wo = df_wo.sort_values('drop', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(df_wo['feature'], df_wo['drop']*100, color='#E67E22', alpha=0.8)
    
    ax.set_xlabel('Performance Drop (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance (Ablation Study)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # valueLabel
    for i, (feature, drop) in enumerate(zip(df_wo['feature'], df_wo['drop'])):
        ax.text(drop*100 + 0.02, i, f'{drop*100:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("alreadyGenerate: ablation_study.png")


def plot_subgroup_analysis():
    """Subgroup analysis"""
    df = pd.read_csv('output/subgroup_analysis.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # byClassGroup
    age_data = df[df['subgroup'].str.contains('Age:')]
    gender_data = df[df['subgroup'].str.contains('Gender:')]
    sofa_data = df[df['subgroup'].str.contains('SOFA:')]
    uo_data = df[df['subgroup'].str.contains('UO:')]
    
    # agegroup
    ax = axes[0, 0]
    age_data['label'] = age_data['subgroup'].str.replace('Age: ', '')
    bars = ax.bar(range(len(age_data)), age_data['adr'], color='#3498DB', alpha=0.8)
    ax.set_xticks(range(len(age_data)))
    ax.set_xticklabels(age_data['label'], rotation=15)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Age Groups', fontweight='bold')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% Target')
    ax.set_ylim(0.85, 0.96)
    for i, v in enumerate(age_data['adr']):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    
    # Sexgroup
    ax = axes[0, 1]
    gender_data['label'] = gender_data['subgroup'].str.replace('Gender: ', '')
    bars = ax.bar(range(len(gender_data)), gender_data['adr'], color='#E74C3C', alpha=0.8)
    ax.set_xticks(range(len(gender_data)))
    ax.set_xticklabels(gender_data['label'])
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Gender Groups', fontweight='bold')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0.85, 0.96)
    for i, v in enumerate(gender_data['adr']):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    
    # SOFA Group
    ax = axes[1, 0]
    sofa_data['label'] = sofa_data['subgroup'].str.replace('SOFA: ', '')
    bars = ax.bar(range(len(sofa_data)), sofa_data['adr'], color='#2ECC71', alpha=0.8)
    ax.set_xticks(range(len(sofa_data)))
    ax.set_xticklabels(sofa_data['label'], rotation=15)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('SOFA Score Groups', fontweight='bold')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0.85, 1.0)
    for i, v in enumerate(sofa_data['adr']):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    
    # Urine outputgroup
    ax = axes[1, 1]
    uo_data['label'] = uo_data['subgroup'].str.replace('UO: Urine output ', '')
    bars = ax.bar(range(len(uo_data)), uo_data['adr'], color='#F39C12', alpha=0.8)
    ax.set_xticks(range(len(uo_data)))
    ax.set_xticklabels(uo_data['label'], rotation=15)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Urine Output Groups', fontweight='bold')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0.85, 1.0)
    for i, v in enumerate(uo_data['adr']):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    
    plt.suptitle('Subgroup Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subgroup_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("alreadyGenerate: subgroup_analysis.png")


def plot_temporal_analysis():
    """Temporal decision pointAnalysis"""
    df = pd.read_csv('output/temporal_analysis.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ADR 
    ax = axes[0]
    timepoints = df['timepoint'].values
    adr = df['adr'].values
    
    ax.plot(range(len(timepoints)), adr, marker='o', linewidth=2, markersize=8, color='#3498DB')
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints)
    ax.set_ylabel('Overall ADR', fontsize=12, fontweight='bold')
    ax.set_xlabel('Decision Point', fontsize=12, fontweight='bold')
    ax.set_title('ADR Across Decision Points', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.0)
    for i, v in enumerate(adr):
        ax.text(i, v + 0.005, f'{v:.2%}', ha='center', fontsize=9)
    
    # TreatmentgroupAccuracy
    ax = axes[1]
    treat_acc = df['treat_acc'].values
    
    ax.bar(range(len(timepoints)), treat_acc, color='#E74C3C', alpha=0.8)
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints)
    ax.set_ylabel('Treatment Group Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Decision Point', fontsize=12, fontweight='bold')
    ax.set_title('Treatment Identification Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(treat_acc):
        ax.text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("alreadyGenerate: temporal_analysis.png")


def plot_bootstrap_ci():
    """BootstrapConfidence interval"""
    df = pd.read_csv('Method /bootstrap_ci.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = df['metric'].values
    means = df['mean'].values
    ci_lower = df['ci_lower'].values
    ci_upper = df['ci_upper'].values
    
    y_pos = np.arange(len(metrics))
    errors = np.array([means - ci_lower, ci_upper - means])
    
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    ax.barh(y_pos, means, xerr=errors, color=colors, alpha=0.8, capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Bootstrap 95% Confidence Intervals (n=1000)', fontsize=14, fontweight='bold')
    
    # valueLabel
    for i, (m, cl, cu) in enumerate(zip(means, ci_lower, ci_upper)):
        ax.text(cu + 0.01, i, f'{m:.2%}\n[{cl:.2%}, {cu:.2%}]', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bootstrap_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("alreadyGenerate: bootstrap_ci.png")


def plot_eicu_threshold():
    """eICUThresholdOptimization"""
    df = pd.read_csv('ExternalValidation/eicu_threshold_optimization.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ADR vs Threshold
    ax = axes[0]
    ax.plot(df['threshold'], df['accuracy'], marker='o', linewidth=2, color='#3498DB', label='Overall ADR')
    ax.plot(df['threshold'], df['treat_acc'], marker='s', linewidth=2, color='#E74C3C', label='RRT Group Acc')
    ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('eICU External Validation: Threshold Optimization', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Threshold 
    best_idx = df['treat_acc'].idxmax()
    best_thresh = df.loc[best_idx, 'threshold']
    best_treat = df.loc[best_idx, 'treat_acc']
    ax.axvline(x=best_thresh, color='green', linestyle='--', alpha=0.5, label=f'Optimal: {best_thresh:.2f}')
    ax.legend()
    
    # Comparison vsOptimization 
    ax = axes[1]
    categories = ['Original\n(thresh=0.5)', 'Optimized\n(thresh=0.10)']
    original_adr = df[df['threshold'] == 0.5]['accuracy'].values[0]
    optimized_adr = df[df['threshold'] == 0.1]['accuracy'].values[0]
    original_treat = df[df['threshold'] == 0.5]['treat_acc'].values[0]
    optimized_treat = df[df['threshold'] == 0.1]['treat_acc'].values[0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [original_adr, optimized_adr], width, 
                   label='Overall ADR', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, [original_treat, optimized_treat], width,
                   label='RRT Group Acc', color='#E74C3C', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Threshold Optimization Effect', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # valueLabel
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eicu_threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("alreadyGenerate: eicu_threshold_optimization.png")


def generate_all_figures():
    """GenerateAll Figure"""
    print("StartGeneratepaperFigure...")
    print("="*50)
    
    plot_baseline_comparison()
    plot_ablation_study()
    plot_subgroup_analysis()
    plot_temporal_analysis()
    plot_bootstrap_ci()
    plot_eicu_threshold()
    
    print("="*50)
    print(f"All FigurealreadySave : {OUTPUT_DIR}")


if __name__ == '__main__':
    generate_all_figures()
