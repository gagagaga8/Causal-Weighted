"""
 Figure: Fig.7 PRCurve + Fig.8 SHAPexplainability
  IEEE JBHI standard & will 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Nature/Science 
PALETTE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import shap
import os
import warnings
warnings.filterwarnings('ignore')

SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]

FEATURE_NAMES = {
    'admission_age': 'Age', 'gender': 'Gender', 'weight': 'Weight',
    'sofa_24hours': 'SOFA Score', 'aki_stage': 'AKI Stage',
    'aki_stage_creat': 'AKI (Creatinine)', 'aki_stage_uo': 'AKI (Urine)',
    'creat': 'Creatinine', 'uo_rt_6hr': 'Urine Output (6h)',
    'uo_rt_12hr': 'Urine Output (12h)', 'uo_rt_24hr': 'Urine Output (24h)',
    'uo_k1': 'UO Trend k1', 'bun_k1': 'BUN Trend k1', 'pot_k1': 'K+ Trend k1',
    'ph_k1': 'pH Trend k1', 'creat_k1': 'Cr Trend k1',
    'uo_k2': 'UO Trend k2', 'bun_k2': 'BUN Trend k2', 'pot_k2': 'K+ Trend k2',
    'ph_k2': 'pH Trend k2', 'creat_k2': 'Cr Trend k2',
    'uo_k3': 'UO Trend k3', 'bun_k3': 'BUN Trend k3', 'pot_k3': 'K+ Trend k3',
    'ph_k3': 'pH Trend k3', 'creat_k3': 'Cr Trend k3',
}

PARAMS = {
    'n_estimators': 100, 'max_depth': 4, 'num_leaves': 15,
    'learning_rate': 0.05, 'min_child_samples': 100,
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

OUTPUT_DIR = 'c:/Dynamic-RRT/figures/'


def load_data():
    """Load data"""
    df_mimic = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
    X_mimic = df_mimic[SAFE_FEATURES].copy()
    X_mimic['gender'] = X_mimic['gender'].map({'M':1,'F':0}).fillna(0)
    X_mimic = X_mimic.fillna(0)
    valid_mask = ~df_mimic['hfd'].isna()
    X_mimic_df = X_mimic[valid_mask]
    X_mimic = X_mimic_df.values
    df_mimic_valid = df_mimic[valid_mask]
    T_mimic = ((df_mimic_valid['a1']==1)|(df_mimic_valid['a2']==1)|(df_mimic_valid['a3']==1)).astype(int).values
    
    df_eicu = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
    df_eicu = df_eicu.drop_duplicates(subset=['patientunitstayid'])
    X_eicu = df_eicu[SAFE_FEATURES].copy()
    X_eicu['gender'] = X_eicu['gender'].map({'M':1,'F':0}).fillna(0)
    X_eicu = X_eicu.fillna(0).values
    T_eicu = df_eicu['received_rrt'].values
    
    return X_mimic, T_mimic, X_eicu, T_eicu, X_mimic_df


def fig7_pr_curves(X_mimic, T_mimic, X_eicu, T_eicu):
    """Fig.7: Precision-RecallCurve"""
    print('Generate Fig.7 PR Curves...')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = [
        ('LightGBM (Ours)', lgb.LGBMClassifier(**PARAMS), PALETTE[0], 2.5),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42), PALETTE[1], 1.5),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), PALETTE[2], 1.5),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), PALETTE[3], 1.5)
    ]
    
    for ax, (X, T, title, panel) in zip(axes, [
        (X_mimic, T_mimic, 'MIMIC-IV (Internal Validation)', 'A'),
        (X_eicu, T_eicu, 'eICU (External Validation)', 'B')
    ]):
        # PartitionTrainingTestset
        X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=42, stratify=T)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Baseline (RandomClassificationer)
        baseline = T_test.mean()
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, alpha=0.6, label=f'Baseline (No Skill)')
        
        for name, model, color, lw in models:
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train_s, T_train)
            prob = model_copy.predict_proba(X_test_s)[:,1]
            
            precision, recall, _ = precision_recall_curve(T_test, prob)
            ap = average_precision_score(T_test, prob)
            ax.plot(recall, precision, color=color, lw=lw, label=f'{name} (AP={ap:.3f})')
        
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig7_PR_Curves.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig7_PR_Curves.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig7_PR_Curves.png/pdf')


def fig8_shap_analysis(X_mimic, T_mimic, X_mimic_df):
    """Fig.8: SHAPexplainabilityAnalysis ( )"""
    print('Generate Fig.8 SHAP Analysis...')
    
    # TrainingModel
    X_train, X_test, T_train, T_test = train_test_split(X_mimic, T_mimic, test_size=0.3, random_state=42, stratify=T_mimic)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train_s, T_train)
    
    # ComputationSHAPvalue
    print(' ComputationSHAPvalue (cancanneedto minutes)...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)
    
    # Processing classOutput
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # classSHAPvalue
    
    # Feature mapping
    feature_names = [FEATURE_NAMES.get(f, f) for f in SAFE_FEATURES]
    
    # Create 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: SHAP Summary Plot (Beeswarm)
    ax1 = axes[0]
    plt.sca(ax1)
    shap.summary_plot(shap_values, X_test_s, feature_names=feature_names, 
                      show=False, max_display=15, plot_size=None)
    ax1.set_title('SHAP Feature Impact (Beeswarm)')
    ax1.set_xlabel('SHAP Value (Impact on RRT Prediction)')
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    
    # Panel B: SHAP Bar Plot (Mean Absolute)
    ax2 = axes[1]
    plt.sca(ax2)
    shap.summary_plot(shap_values, X_test_s, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=15, plot_size=None)
    ax2.set_title('Mean |SHAP Value| (Feature Importance)')
    ax2.set_xlabel('Mean |SHAP Value|')
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig8_SHAP_Analysis.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig8_SHAP_Analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig8_SHAP_Analysis.png/pdf')
    
    # Generate SHAP Dependence Plot (Top 3Feature)
    print('Generate Fig.9 SHAP Dependence Plots...')
    
    # ComputationFeature to Sort
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for idx, (ax, feat_idx) in enumerate(zip(axes, top_indices)):
        plt.sca(ax)
        shap.dependence_plot(feat_idx, shap_values, X_test_s, 
                            feature_names=feature_names, show=False, ax=ax)
        ax.set_title(f'SHAP Dependence: {feature_names[feat_idx]}')
        panel_label = chr(65 + idx)  # A, B, C
        ax.text(-0.15, 1.08, panel_label, transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig9_SHAP_Dependence.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig9_SHAP_Dependence.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig9_SHAP_Dependence.png/pdf')


def main():
    print('='*70)
    print(' Figure: PRCurve + SHAPexplainability')
    print('='*70)
    
    # Load data
    X_mimic, T_mimic, X_eicu, T_eicu, X_mimic_df = load_data()
    print(f'\nMIMIC: {len(X_mimic)} samples, RRT rate={T_mimic.mean():.2%}')
    print(f'eICU:  {len(X_eicu)} samples, RRT rate={T_eicu.mean():.2%}')
    
    # GenerateFigure
    print('\n' + '='*70)
    print('Generate Figure')
    print('='*70)
    
    fig7_pr_curves(X_mimic, T_mimic, X_eicu, T_eicu)
    fig8_shap_analysis(X_mimic, T_mimic, X_mimic_df)
    
    # Summary
    print('\n' + '='*70)
    print('Complete')
    print('='*70)
    print('\n Figure:')
    print('  Fig.7  PR Curves (Precision-Recall)')
    print('  Fig.8  SHAP Analysis (Beeswarm + Bar)')
    print('  Fig.9  SHAP Dependence (Top 3 Features)')
    print(f'\nSave : {OUTPUT_DIR}')
    
    # Column All Figure
    print('\nwhen Full Figure (Fig.1-9):')
    import glob
    files = sorted(glob.glob(f'{OUTPUT_DIR}Fig*.png'))
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(f) / 1024
        print(f'  {name} ({size:.0f} KB)')


if __name__ == '__main__':
    main()
