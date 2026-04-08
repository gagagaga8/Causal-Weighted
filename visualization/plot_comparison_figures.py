"""
Comparison experiment visualization
Per user request: increase in-figure font size by 3x
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Font size settings (3x larger)
TITLE_SIZE = 42
LABEL_SIZE = 36
TICK_SIZE = 30
LEGEND_SIZE = 27

def plot_sampling_comparison():
    """SamplingpolicyComparison """
    df = pd.read_csv('Experiment/sampling_comparison.csv')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    strategies = df['strategy'].values
    adr = df['adr'].values * 100
    recall = df['rrt_recall'].values * 100
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, adr, width, label='ADR', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, recall, width, label='RRT Recall', color='coral', alpha=0.8)
    
    ax.set_xlabel('Sampling Strategy', fontsize=LABEL_SIZE)
    ax.set_ylabel('Performance (%)', fontsize=LABEL_SIZE)
    ax.set_title('Sampling Strategy Comparison', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' ', '\n') for s in strategies], fontsize=TICK_SIZE-6, rotation=0)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax.set_ylim(0, 100)
    ax.axhline(y=93.06, color='red', linestyle='--', linewidth=2, label='Baseline ADR')
    ax.grid(axis='y', alpha=0.3)
    
    # valueLabel
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=TICK_SIZE-6)
    
    plt.tight_layout()
    plt.savefig('Visualization/figures/sampling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ sampling_comparison.png")

def plot_ensemble_comparison():
    """set policyComparison """
    df = pd.read_csv('Experiment/ensemble_comparison.csv')
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    strategies = df['strategy'].values
    adr = df['adr'].values * 100
    std = df['adr_std'].values * 100
    
    colors = ['gold' if s == 'Single LightGBM' else 'steelblue' for s in strategies]
    
    bars = ax.barh(strategies, adr, xerr=std, color=colors, alpha=0.8, capsize=5)
    
    ax.set_xlabel('ADR (%)', fontsize=LABEL_SIZE)
    ax.set_title('Ensemble Strategy Comparison', fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_xlim(80, 96)
    ax.axvline(x=93.06, color='red', linestyle='--', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    # valueLabel
    for bar, val in zip(bars, adr):
        ax.annotate(f'{val:.2f}%', xy=(val, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=TICK_SIZE-3)
    
    plt.tight_layout()
    plt.savefig('Visualization/figures/ensemble_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ ensemble_comparison.png")

def plot_threshold_analysis():
    """Threshold Analysis """
    df = pd.read_csv('Experiment/threshold_analysis.csv')
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    thresholds = df['threshold'].values
    accuracy = df['accuracy'].values * 100
    recall = df['recall'].values * 100
    specificity = df['specificity'].values * 100
    f1 = df['f1'].values * 100
    
    ax.plot(thresholds, accuracy, 'o-', label='Accuracy', linewidth=3, markersize=12, color='steelblue')
    ax.plot(thresholds, recall, 's-', label='Recall (RRT)', linewidth=3, markersize=12, color='coral')
    ax.plot(thresholds, specificity, '^-', label='Specificity', linewidth=3, markersize=12, color='green')
    ax.plot(thresholds, f1, 'd-', label='F1 Score', linewidth=3, markersize=12, color='purple')
    
    ax.set_xlabel('Decision Threshold', fontsize=LABEL_SIZE)
    ax.set_ylabel('Performance (%)', fontsize=LABEL_SIZE)
    ax.set_title('Threshold Sensitivity Analysis', fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='center right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('Visualization/figures/threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ threshold_analysis.png")

def plot_learning_curve():
    """ Curve """
    df = pd.read_csv('Experiment/learning_curve.csv')
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    ratios = df['train_ratio'].values * 100
    adr = df['adr'].values * 100
    std = df['adr_std'].values * 100
    
    ax.errorbar(ratios, adr, yerr=std, fmt='o-', linewidth=3, markersize=12, 
                capsize=5, color='steelblue', ecolor='coral')
    ax.fill_between(ratios, adr - std, adr + std, alpha=0.2, color='steelblue')
    
    ax.set_xlabel('Training Data Ratio (%)', fontsize=LABEL_SIZE)
    ax.set_ylabel('ADR (%)', fontsize=LABEL_SIZE)
    ax.set_title('Learning Curve Analysis', fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_xlim(0, 105)
    ax.set_ylim(90, 96)
    ax.axhline(y=93.06, color='red', linestyle='--', linewidth=2, label='Full Data ADR')
    ax.axhline(y=93.06*0.95, color='orange', linestyle=':', linewidth=2, label='95% Performance')
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Visualization/figures/learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ learning_curve.png")

def plot_feature_selection():
    """Feature Comparison """
    df = pd.read_csv('Experiment/feature_selection_comparison.csv')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    methods = df['method'].values
    n_features = df['n_features'].values
    adr = df['adr'].values * 100
    
    # byFeaturesGroup 
    colors = []
    for n in n_features:
        if n == 26:
            colors.append('gold')
        elif n == 20:
            colors.append('steelblue')
        elif n == 15:
            colors.append('coral')
        else:
            colors.append('green')
    
    bars = ax.barh(range(len(methods)), adr, color=colors, alpha=0.8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=TICK_SIZE-6)
    ax.set_xlabel('ADR (%)', fontsize=LABEL_SIZE)
    ax.set_title('Feature Selection Strategy Comparison', fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.set_xlim(90, 94)
    ax.axvline(x=93.06, color='red', linestyle='--', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gold', label='All Features (26)'),
                      Patch(facecolor='steelblue', label='k=20'),
                      Patch(facecolor='coral', label='k=15'),
                      Patch(facecolor='green', label='k=10')]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE-3, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('Visualization/figures/feature_selection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ feature_selection.png")

def plot_timewindow_sensitivity():
    """ """
    df = pd.read_csv('Experiment/timewindow_sensitivity.csv')
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    configs = df['config'].values
    adr = df['adr'].values * 100
    n_features = df['n_features'].values
    
    colors = ['gold' if 'Full' in c else 'steelblue' for c in configs]
    
    bars = ax.bar(range(len(configs)), adr, color=colors, alpha=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=TICK_SIZE-6, rotation=0)
    ax.set_ylabel('ADR (%)', fontsize=LABEL_SIZE)
    ax.set_title('Time Window Configuration Comparison', fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_ylim(88, 95)
    ax.axhline(y=93.06, color='red', linestyle='--', linewidth=2)
    ax.grid(axis='y', alpha=0.3)
    
    # FeaturesLabel
    for bar, nf, val in zip(bars, n_features, adr):
        ax.annotate(f'{val:.1f}%\n({nf})', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=TICK_SIZE-6)
    
    plt.tight_layout()
    plt.savefig('Visualization/figures/timewindow_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ timewindow_sensitivity.png")

if __name__ == '__main__':
    print("GenerateComparisonExperimentFigure...")
    plot_sampling_comparison()
    plot_ensemble_comparison()
    plot_threshold_analysis()
    plot_learning_curve()
    plot_feature_selection()
    plot_timewindow_sensitivity()
    print("\nall FigureGenerateComplete ")
