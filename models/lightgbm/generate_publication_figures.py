"""
IEEE JBHI FigureGenerate
 : NeurIPS/ICML/Nature/Science standard
- DPI Output (600 DPI)
- Nature scheme
- Arial/Helvetica 
- Fig.1 - Fig.6
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ========================
# all settings
# ========================
# Nature/Science scheme
COLORS = {
    'primary': '#4C72B0', # replaceas and 
    'secondary': '#DD8452', # replaceas and 
    'tertiary': '#55A868', # replaceas and 
    'quaternary': '#C44E52', # replaceas 
    'quinary': '#8172B3', # replaceas and 
    'senary': '#937860', # replaceas 
    'gray': '#7E6148',
    'black': '#000000',
}
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

# willstandard 
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
    'savefig.pad_inches': 0.1,
})

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, 
                           precision_recall_curve, average_precision_score,
                           confusion_matrix, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from scipy import stats
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

PARAMS = {
    'n_estimators': 100, 'max_depth': 4, 'num_leaves': 15,
    'learning_rate': 0.05, 'min_child_samples': 100,
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

OUTPUT_DIR = 'c:/Dynamic-RRT/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load data"""
    df_mimic = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
    X_mimic = df_mimic[SAFE_FEATURES].copy()
    X_mimic['gender'] = X_mimic['gender'].map({'M':1,'F':0}).fillna(0)
    X_mimic = X_mimic.fillna(0)
    valid_mask = ~df_mimic['hfd'].isna()
    X_mimic = X_mimic[valid_mask].values
    df_mimic_valid = df_mimic[valid_mask]
    T_mimic = ((df_mimic_valid['a1']==1)|(df_mimic_valid['a2']==1)|(df_mimic_valid['a3']==1)).astype(int).values
    
    df_eicu = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
    df_eicu = df_eicu.drop_duplicates(subset=['patientunitstayid'])
    X_eicu = df_eicu[SAFE_FEATURES].copy()
    X_eicu['gender'] = X_eicu['gender'].map({'M':1,'F':0}).fillna(0)
    X_eicu = X_eicu.fillna(0).values
    T_eicu = df_eicu['received_rrt'].values
    
    return X_mimic, T_mimic, X_eicu, T_eicu, df_mimic[valid_mask]


def run_statistical_tests(X, T):
    """statisticsSignificanceTest"""
    print('\nRunstatisticsSignificanceTest (10-fold ×3repeatCV)...')
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    results = {'lgb': [], 'lr': [], 'rf': [], 'gb': []}
    
    for train_idx, test_idx in rskf.split(X, T):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        for name, model in [
            ('lgb', lgb.LGBMClassifier(**PARAMS)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
        ]:
            model.fit(X_train_s, T_train)
            results[name].append(roc_auc_score(T_test, model.predict_proba(X_test_s)[:,1]))
    
    # WilcoxonTest
    _, p_lr = stats.wilcoxon(results['lgb'], results['lr'], alternative='greater')
    _, p_rf = stats.wilcoxon(results['lgb'], results['rf'])
    
    print(f'  LightGBM: {np.mean(results["lgb"]):.4f} ± {np.std(results["lgb"]):.4f}')
    print(f'  vs LR: p={p_lr:.4f} {"***" if p_lr<0.001 else "**" if p_lr<0.01 else "*" if p_lr<0.05 else ""}')
    
    return results, p_lr


def fig1_roc_curves(X_mimic, T_mimic, X_eicu, T_eicu):
    """Fig.1: ROCCurveComparison ( )"""
    print('Generate Fig.1 ROC Curves...')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = [
        ('LightGBM (Ours)', lgb.LGBMClassifier(**PARAMS), PALETTE[0]),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42), PALETTE[1]),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), PALETTE[2]),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), PALETTE[3])
    ]
    
    for ax, (X, T, title, panel) in zip(axes, [
        (X_mimic, T_mimic, 'MIMIC-IV (Internal Validation)', 'A'),
        (X_eicu, T_eicu, 'eICU (External Validation)', 'B')
    ]):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for name, model, color in models:
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_scaled, T)
            prob = model_copy.predict_proba(X_scaled)[:,1]
            fpr, tpr, _ = roc_curve(T, prob)
            auc = roc_auc_score(T, prob)
            lw = 2.5 if 'Ours' in name else 1.5
            ax.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC={auc:.3f})')
        
        ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.4, label='Random Chance')
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig1_ROC_Curves.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig1_ROC_Curves.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig1_ROC_Curves.png/pdf')


def fig2_calibration(X_mimic, T_mimic, X_eicu, T_eicu):
    """Fig.2: Calibration Curve"""
    print('Generate Fig.2 Calibration Curves...')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (X, T, title, panel) in zip(axes, [
        (X_mimic, T_mimic, 'MIMIC-IV', 'A'),
        (X_eicu, T_eicu, 'eICU', 'B')
    ]):
        X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=42, stratify=T)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_train_s, T_train)
        prob = model.predict_proba(X_test_s)[:,1]
        
        fraction_pos, mean_pred = calibration_curve(T_test, prob, n_bins=10)
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Perfectly Calibrated')
        ax.plot(mean_pred, fraction_pos, 's-', color=PALETTE[0], lw=2, markersize=8, 
                markeredgecolor='white', markeredgewidth=1.5, label='LightGBM')
        
        # fillConfidence interval 
        ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - {title}')
        ax.legend(loc='lower right')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig2_Calibration.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig2_Calibration.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig2_Calibration.png/pdf')


def fig3_model_comparison(stat_results, p_lr):
    """Fig.3: ModelComparison records """
    print('Generate Fig.3 Model Comparison...')
    
    models = ['LightGBM\n(Ours)', 'Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
    aucs = [np.mean(stat_results[k]) for k in ['lgb', 'lr', 'rf', 'gb']]
    stds = [np.std(stat_results[k]) for k in ['lgb', 'lr', 'rf', 'gb']]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, aucs, yerr=stds, capsize=4, color=PALETTE[:4], 
                  edgecolor='black', linewidth=1, alpha=0.85)
    
    # Model
    bars[0].set_edgecolor(PALETTE[0])
    bars[0].set_linewidth(2.5)
    
    ax.set_ylabel('AUC-ROC')
    ax.set_xlabel('Model')
    ax.set_title('Model Performance Comparison\n(10-fold × 3 Repeated Cross-Validation)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.78, 0.88)
    
    # valueLabel
    for i, (v, s) in enumerate(zip(aucs, stds)):
        ax.text(i, v + s + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Significance 
    if p_lr < 0.001:
        sig_text = '***'
    elif p_lr < 0.01:
        sig_text = '**'
    elif p_lr < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    
    y_max = max(aucs[0], aucs[1]) + max(stds[0], stds[1]) + 0.02
    ax.plot([0, 1], [y_max, y_max], 'k-', lw=1)
    ax.text(0.5, y_max + 0.005, sig_text, ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig3_Model_Comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig3_Model_Comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig3_Model_Comparison.png/pdf')


def fig4_feature_importance(X, T):
    """Fig.4: Feature to """
    print('Generate Fig.4 Feature Importance...')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_scaled, T)
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]
    
    # Feature 
    feature_names_map = {
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
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    y_pos = np.arange(15)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 15))
    
    bars = ax.barh(y_pos, importance[indices][::-1], color=colors[::-1], 
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_yticks(y_pos)
    labels = [feature_names_map.get(SAFE_FEATURES[i], SAFE_FEATURES[i]) for i in indices][::-1]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title('Top 15 Feature Importance')
    
    # valueLabel
    for i, v in enumerate(importance[indices][::-1]):
        ax.text(v + 1, i, f'{v:.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig4_Feature_Importance.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig4_Feature_Importance.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig4_Feature_Importance.png/pdf')


def fig5_ablation(X, T, df):
    """Fig.5: AblationExperiment"""
    print('Generate Fig.5 Ablation Study...')
    
    feature_groups = {
        'Full Model (26 features)': SAFE_FEATURES,
        'w/o Demographics': [f for f in SAFE_FEATURES if f not in ['admission_age', 'gender', 'weight']],
        'w/o SOFA & AKI': [f for f in SAFE_FEATURES if f not in ['sofa_24hours', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo']],
        'w/o Urine Output': [f for f in SAFE_FEATURES if 'uo' not in f],
        'w/o Lab Trends': [f for f in SAFE_FEATURES if f not in ['bun_k1','bun_k2','bun_k3','pot_k1','pot_k2','pot_k3','ph_k1','ph_k2','ph_k3']],
        'w/o Timepoints k2,k3': [f for f in SAFE_FEATURES if 'k2' not in f and 'k3' not in f],
    }
    
    results = []
    for group_name, features in feature_groups.items():
        available = [f for f in features if f in df.columns]
        X_sub = df[available].copy()
        if 'gender' in X_sub.columns:
            X_sub['gender'] = X_sub['gender'].map({'M':1,'F':0}).fillna(0)
        X_sub = X_sub.fillna(0).values
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for train_idx, test_idx in skf.split(X_sub, T):
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_sub[train_idx])
            X_test_s = scaler.transform(X_sub[test_idx])
            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(X_train_s, T[train_idx])
            aucs.append(roc_auc_score(T[test_idx], model.predict_proba(X_test_s)[:,1]))
        
        results.append({'config': group_name, 'auc': np.mean(aucs), 'std': np.std(aucs)})
    
    df_results = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(df_results))
    colors = [PALETTE[2]] + [PALETTE[0]]*5 # as asAblation
    
    bars = ax.barh(y_pos, df_results['auc'], xerr=df_results['std'], 
                   color=colors, capsize=4, edgecolor='black', linewidth=0.8, alpha=0.85)
    
    ax.axvline(x=df_results.iloc[0]['auc'], color=PALETTE[2], linestyle='--', lw=1.5, alpha=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_results['config'])
    ax.set_xlabel('AUC-ROC')
    ax.set_title('Ablation Study: Feature Group Contribution')
    ax.set_xlim(0.80, 0.88)
    
    # valueand valueLabel
    baseline = df_results.iloc[0]['auc']
    for i, (v, s) in enumerate(zip(df_results['auc'], df_results['std'])):
        diff = v - baseline
        diff_str = f' ({diff:+.3f})' if i > 0 else ''
        ax.text(v + s + 0.002, i, f'{v:.3f}{diff_str}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig5_Ablation_Study.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig5_Ablation_Study.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig5_Ablation_Study.png/pdf')


def fig6_confusion_matrix(X_mimic, T_mimic, X_eicu, T_eicu):
    """Fig.6: confusion matrix"""
    print('Generate Fig.6 Confusion Matrix...')
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    for ax, (X, T, title, panel) in zip(axes, [
        (X_mimic, T_mimic, 'MIMIC-IV (Internal)', 'A'),
        (X_eicu, T_eicu, 'eICU (External)', 'B')
    ]):
        X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=42, stratify=T)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_train_s, T_train)
        pred = model.predict(X_test_s)
        
        cm = confusion_matrix(T_test, pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        ax.set_title(f'Confusion Matrix - {title}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No RRT', 'RRT'])
        ax.set_yticklabels(['No RRT', 'RRT'])
        
        for i in range(2):
            for j in range(2):
                text = f'{cm[i,j]}\n({cm_norm[i,j]:.1%})'
                color = 'white' if cm_norm[i,j] > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=12, fontweight='bold')
        
        ax.text(-0.25, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig6_Confusion_Matrix.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}Fig6_Confusion_Matrix.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Save: Fig6_Confusion_Matrix.png/pdf')


def main():
    print('='*70)
    print('IEEE JBHI FigureGenerate')
    print(' standard: NeurIPS/ICML/Nature/Science')
    print('='*70)
    
    # Load data
    X_mimic, T_mimic, X_eicu, T_eicu, df_mimic = load_data()
    print(f'\nMIMIC: {len(X_mimic)} samples, RRT rate={T_mimic.mean():.2%}')
    print(f'eICU:  {len(X_eicu)} samples, RRT rate={T_eicu.mean():.2%}')
    
    # statisticsTest
    stat_results, p_lr = run_statistical_tests(X_mimic, T_mimic)
    
    # Generate6 Figure
    print('\n' + '='*70)
    print('Generate Figure (600 DPI, Nature )')
    print('='*70)
    
    fig1_roc_curves(X_mimic, T_mimic, X_eicu, T_eicu)
    fig2_calibration(X_mimic, T_mimic, X_eicu, T_eicu)
    fig3_model_comparison(stat_results, p_lr)
    fig4_feature_importance(X_mimic, T_mimic)
    fig5_ablation(X_mimic, T_mimic, df_mimic)
    fig6_confusion_matrix(X_mimic, T_mimic, X_eicu, T_eicu)
    
    # Figure
    import glob
    old_files = glob.glob(f'{OUTPUT_DIR}fig_*.png')
    for f in old_files:
        os.remove(f)
        print(f' Delete Figure: {os.path.basename(f)}')
    
    # Summary
    print('\n' + '='*70)
    print('GenerateComplete')
    print('='*70)
    print(f'\nSave : {OUTPUT_DIR}')
    print('\nFigure :')
    print('  Fig.1  ROC Curves (Internal & External Validation)')
    print('  Fig.2  Calibration Curves')
    print('  Fig.3  Model Performance Comparison')
    print('  Fig.4  Feature Importance (Top 15)')
    print('  Fig.5  Ablation Study')
    print('  Fig.6  Confusion Matrix')
    print('\n : PNG (600 DPI) + PDF ( )')
    print(' : Nature/Sciencestandard ')
    print(' : Arial/Helvetica')


if __name__ == '__main__':
    main()
