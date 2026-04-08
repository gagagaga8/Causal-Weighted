"""
 useValidation processed Data GenerateAll Figure
 FigureDatawithpaper value all 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                           roc_auc_score, confusion_matrix, brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

# willstandardConfiguration - Optimization Size
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14, # 
    'axes.labelsize': 16, # Label
    'axes.titlesize': 18, # title
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 13, # 
    'xtick.labelsize': 13, # Label
    'ytick.labelsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
OUTPUT_DIR = 'c:/Dynamic-RRT/figures/'

FEATURES = [
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
    'random_state': 42, 'verbose': -1
}


def load_data():
    """LoadingAll Data"""
    # MIMIC
    df = pd.read_csv('../dWOLS/data/dwols_full_with_uo.csv')
    X = df[FEATURES].copy()
    X['gender'] = X['gender'].map({'M':1,'F':0}).fillna(0)
    X = X.fillna(0)
    mask = ~df['hfd'].isna()
    X_mimic = X[mask].values
    T_mimic = ((df[mask]['a1']==1)|(df[mask]['a2']==1)|(df[mask]['a3']==1)).astype(int).values
    
    # eICU
    df_e = pd.read_csv('../../2_eICUPreprocessingdata/eicu_full_features.csv')
    df_e = df_e.drop_duplicates(subset=['patientunitstayid'])
    X_e = df_e[FEATURES].copy()
    X_e['gender'] = X_e['gender'].map({'M':1,'F':0}).fillna(0)
    X_eicu = X_e.fillna(0).values
    T_eicu = df_e['received_rrt'].values
    
    # transfer learningPartition
    np.random.seed(42)
    idx = np.random.permutation(len(X_eicu))
    split = len(idx) // 2
    X_adapt = X_eicu[idx[:split]]
    T_adapt = T_eicu[idx[:split]]
    X_test = X_eicu[idx[split:]]
    T_test = T_eicu[idx[split:]]
    
    return X_mimic, T_mimic, X_adapt, T_adapt, X_test, T_test


def fig1_roc_curves():
    """Fig 1: ROCCurve"""
    print('Generate Fig1_ROC_Curves...')
    X_mimic, T_mimic, X_adapt, T_adapt, X_test, T_test = load_data()
    
    # Internal ValidationROC (MIMIC CV)
    scaler1 = StandardScaler()
    X_mimic_s = scaler1.fit_transform(X_mimic)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    for train_idx, test_idx in skf.split(X_mimic_s, T_mimic):
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_mimic_s[train_idx], T_mimic[train_idx])
        prob = model.predict_proba(X_mimic_s[test_idx])[:,1]
        fpr, tpr, _ = roc_curve(T_mimic[test_idx], prob)
        tprs.append(np.interp(all_fpr, fpr, tpr))
        aucs.append(roc_auc_score(T_mimic[test_idx], prob))
    
    mean_tpr_internal = np.mean(tprs, axis=0)
    mean_auc_internal = np.mean(aucs)
    std_auc_internal = np.std(aucs)
    
    # ExternalValidationROC (transfer learning)
    X_combined = np.vstack([X_mimic, X_adapt])
    T_combined = np.hstack([T_mimic, T_adapt])
    scaler2 = StandardScaler()
    X_train = scaler2.fit_transform(X_combined)
    X_test_s = scaler2.transform(X_test)
    
    model_ext = lgb.LGBMClassifier(**PARAMS)
    model_ext.fit(X_train, T_combined)
    prob_ext = model_ext.predict_proba(X_test_s)[:,1]
    fpr_ext, tpr_ext, _ = roc_curve(T_test, prob_ext)
    auc_ext = roc_auc_score(T_test, prob_ext)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (A) Internal Validation
    axes[0].plot(all_fpr, mean_tpr_internal, color=PALETTE[0], lw=3,
                label=f'LightGBM (AUC = {mean_auc_internal:.3f} ± {std_auc_internal:.3f})')
    axes[0].fill_between(all_fpr, 
                         np.mean(tprs, axis=0) - np.std(tprs, axis=0),
                         np.mean(tprs, axis=0) + np.std(tprs, axis=0),
                         alpha=0.2, color=PALETTE[0])
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    axes[0].set_xlabel('False Positive Rate', fontsize=15)
    axes[0].set_ylabel('True Positive Rate', fontsize=15)
    axes[0].set_title('(A) Internal Validation (MIMIC-IV)', fontsize=16, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=12, framealpha=0.9)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    # AUC value in in 
    axes[0].text(0.55, 0.25, f'AUC = {mean_auc_internal:.3f}', fontsize=18, fontweight='bold',
                transform=axes[0].transAxes, color=PALETTE[0])
    
    # (B) ExternalValidation
    axes[1].plot(fpr_ext, tpr_ext, color=PALETTE[1], lw=3,
                label=f'LightGBM (AUC = {auc_ext:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    axes[1].set_xlabel('False Positive Rate', fontsize=15)
    axes[1].set_ylabel('True Positive Rate', fontsize=15)
    axes[1].set_title('(B) External Validation (eICU)', fontsize=16, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=12, framealpha=0.9)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    # AUC value in in 
    axes[1].text(0.55, 0.25, f'AUC = {auc_ext:.3f}', fontsize=18, fontweight='bold',
                transform=axes[1].transAxes, color=PALETTE[1])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig1_ROC_Curves.png', dpi=600)
    plt.savefig(f'{OUTPUT_DIR}Fig1_ROC_Curves.pdf')
    plt.close()
    
    print(f'  Internal AUC: {mean_auc_internal:.4f} ± {std_auc_internal:.4f}')
    print(f'  ExternalAUC: {auc_ext:.4f}')
    return {'internal_auc': mean_auc_internal, 'external_auc': auc_ext}


def fig2_calibration():
    """Fig 2: Calibration Curve"""
    print('Generate Fig2_Calibration...')
    X_mimic, T_mimic, X_adapt, T_adapt, X_test, T_test = load_data()
    
    # TrainingModel
    X_combined = np.vstack([X_mimic, X_adapt])
    T_combined = np.hstack([T_mimic, T_adapt])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_combined)
    X_test_s = scaler.transform(X_test)
    
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train, T_combined)
    prob = model.predict_proba(X_test_s)[:,1]
    
    # Calibration CurveComputation
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed = []
    predicted = []
    counts = []
    
    for i in range(n_bins):
        mask = (prob >= bin_edges[i]) & (prob < bin_edges[i+1])
        if mask.sum() > 0:
            observed.append(T_test[mask].mean())
            predicted.append(prob[mask].mean())
            counts.append(mask.sum())
        else:
            observed.append(np.nan)
            predicted.append(bin_centers[i])
            counts.append(0)
    
    brier = brier_score_loss(T_test, prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    ax.scatter(predicted, observed, c=PALETTE[0], s=150, zorder=5, edgecolors='white', linewidths=1.5)
    ax.plot(predicted, observed, color=PALETTE[0], lw=2.5, 
           label=f'LightGBM (Brier = {brier:.3f})')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=15)
    ax.set_ylabel('Observed Proportion', fontsize=15)
    ax.set_title('Calibration Curve (eICU External Validation)', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    # Brier value 
    ax.text(0.05, 0.92, f'Brier Score = {brier:.3f}', fontsize=16, fontweight='bold',
           transform=ax.transAxes, color=PALETTE[0])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig2_Calibration.png', dpi=600)
    plt.savefig(f'{OUTPUT_DIR}Fig2_Calibration.pdf')
    plt.close()
    
    print(f'  Brier Score: {brier:.4f}')
    return {'brier': brier}


def fig3_model_comparison():
    """Fig 3: ModelComparison"""
    print('Generate Fig3_Model_Comparison...')
    X_mimic, T_mimic, _, _, _, _ = load_data()
    
    models = {
        'LightGBM': lgb.LGBMClassifier(**PARAMS),
        'Logistic\nRegression': LogisticRegression(max_iter=1000, random_state=42),
        'Random\nForest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Gradient\nBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_mimic)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {name: [] for name in models}
    
    for train_idx, test_idx in skf.split(X_s, T_mimic):
        for name, model in models.items():
            m = model.__class__(**model.get_params())
            m.fit(X_s[train_idx], T_mimic[train_idx])
            prob = m.predict_proba(X_s[test_idx])[:,1]
            results[name].append(roc_auc_score(T_mimic[test_idx], prob))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    positions = range(len(models))
    bp = ax.boxplot([results[name] for name in models.keys()],
                    positions=positions,
                    patch_artist=True,
                    widths=0.6)
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(models.keys())
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model Comparison (10-fold CV)')
    ax.set_ylim(0.7, 1.0)
    
    # value - 
    for i, name in enumerate(models.keys()):
        mean_val = np.mean(results[name])
        ax.annotate(f'{mean_val:.3f}', xy=(i, mean_val + 0.015), 
                   ha='center', fontsize=14, fontweight='bold', color='#333333')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig3_Model_Comparison.png', dpi=600)
    plt.savefig(f'{OUTPUT_DIR}Fig3_Model_Comparison.pdf')
    plt.close()
    
    for name in models:
        print(f'  {name}: {np.mean(results[name]):.4f} ± {np.std(results[name]):.4f}')
    return results


def fig7_pr_curves():
    """Fig 7: PRCurve"""
    print('Generate Fig7_PR_Curves...')
    X_mimic, T_mimic, X_adapt, T_adapt, X_test, T_test = load_data()
    
    # Internal Validation
    scaler1 = StandardScaler()
    X_mimic_s = scaler1.fit_transform(X_mimic)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_recall = np.linspace(0, 1, 100)
    precisions = []
    
    for train_idx, test_idx in skf.split(X_mimic_s, T_mimic):
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_mimic_s[train_idx], T_mimic[train_idx])
        prob = model.predict_proba(X_mimic_s[test_idx])[:,1]
        prec, rec, _ = precision_recall_curve(T_mimic[test_idx], prob)
        precisions.append(np.interp(all_recall, rec[::-1], prec[::-1]))
    
    mean_precision = np.mean(precisions, axis=0)
    
    # ExternalValidation
    X_combined = np.vstack([X_mimic, X_adapt])
    T_combined = np.hstack([T_mimic, T_adapt])
    scaler2 = StandardScaler()
    X_train = scaler2.fit_transform(X_combined)
    X_test_s = scaler2.transform(X_test)
    
    model_ext = lgb.LGBMClassifier(**PARAMS)
    model_ext.fit(X_train, T_combined)
    prob_ext = model_ext.predict_proba(X_test_s)[:,1]
    prec_ext, rec_ext, _ = precision_recall_curve(T_test, prob_ext)
    pr_auc_ext = auc(rec_ext, prec_ext)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Internal ValidationPRCurve
    pr_auc_internal = auc(all_recall, mean_precision)
    axes[0].plot(all_recall, mean_precision, color=PALETTE[0], lw=3,
                label=f'LightGBM (PR-AUC = {pr_auc_internal:.3f})')
    axes[0].fill_between(all_recall, 
                         np.mean(precisions, axis=0) - np.std(precisions, axis=0),
                         np.mean(precisions, axis=0) + np.std(precisions, axis=0),
                         alpha=0.2, color=PALETTE[0])
    axes[0].axhline(y=T_mimic.mean(), color='gray', linestyle='--', lw=1.5, label=f'Baseline ({T_mimic.mean():.2%})')
    axes[0].set_xlabel('Recall', fontsize=15)
    axes[0].set_ylabel('Precision', fontsize=15)
    axes[0].set_title('(A) Internal Validation (MIMIC-IV)', fontsize=16, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=12)
    # value 
    axes[0].text(0.55, 0.75, f'PR-AUC = {pr_auc_internal:.3f}', fontsize=16, fontweight='bold',
                transform=axes[0].transAxes, color=PALETTE[0])
    
    axes[1].plot(rec_ext, prec_ext, color=PALETTE[1], lw=3,
                label=f'PR-AUC = {pr_auc_ext:.3f}')
    axes[1].axhline(y=T_test.mean(), color='gray', linestyle='--', lw=1.5, label=f'Baseline ({T_test.mean():.2%})')
    axes[1].set_xlabel('Recall', fontsize=15)
    axes[1].set_ylabel('Precision', fontsize=15)
    axes[1].set_title('(B) External Validation (eICU)', fontsize=16, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=12)
    # value 
    axes[1].text(0.55, 0.75, f'PR-AUC = {pr_auc_ext:.3f}', fontsize=16, fontweight='bold',
                transform=axes[1].transAxes, color=PALETTE[1])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig7_PR_Curves.png', dpi=600)
    plt.savefig(f'{OUTPUT_DIR}Fig7_PR_Curves.pdf')
    plt.close()
    
    print(f'  ExternalPR-AUC: {pr_auc_ext:.4f}')


def fig6_confusion_matrix():
    """Fig 6: confusion matrix"""
    print('Generate Fig6_Confusion_Matrix...')
    X_mimic, T_mimic, X_adapt, T_adapt, X_test, T_test = load_data()
    
    X_combined = np.vstack([X_mimic, X_adapt])
    T_combined = np.hstack([T_mimic, T_adapt])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_combined)
    X_test_s = scaler.transform(X_test)
    
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train, T_combined)
    pred = model.predict(X_test_s)
    
    cm = confusion_matrix(T_test, pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # pairvalue - 
    im1 = axes[0].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            # pair use nonpair use 
            color = 'white' if (i==j and cm[i,j] > cm.max()*0.3) else 'black'
            axes[0].text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=18, fontweight='bold', color=color)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['No RRT', 'RRT'], fontsize=13)
    axes[0].set_yticklabels(['No RRT', 'RRT'], fontsize=13)
    axes[0].set_xlabel('Predicted', fontsize=15)
    axes[0].set_ylabel('Actual', fontsize=15)
    axes[0].set_title('(A) Absolute Values', fontsize=16, fontweight='bold')
    
    # Percentage - 
    im2 = axes[1].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i,j] > 0.5 else 'black'
            axes[1].text(j, i, f'{cm_norm[i,j]:.1%}', ha='center', va='center', fontsize=18, fontweight='bold', color=color)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['No RRT', 'RRT'], fontsize=13)
    axes[1].set_yticklabels(['No RRT', 'RRT'], fontsize=13)
    axes[1].set_xlabel('Predicted', fontsize=15)
    axes[1].set_ylabel('Actual', fontsize=15)
    axes[1].set_title('(B) Normalized', fontsize=16, fontweight='bold')
    
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}Fig6_Confusion_Matrix.png', dpi=600)
    plt.savefig(f'{OUTPUT_DIR}Fig6_Confusion_Matrix.pdf')
    plt.close()
    
    print(f'  TN={cm[0,0]:,}, FP={cm[0,1]:,}, FN={cm[1,0]:,}, TP={cm[1,1]:,}')


def main():
    print('='*70)
    print(' GenerateAll Figure - useValidation processed Data')
    print('='*70)
    
    # Generate Figure
    roc_data = fig1_roc_curves()
    cal_data = fig2_calibration()
    comp_data = fig3_model_comparison()
    fig6_confusion_matrix()
    fig7_pr_curves()
    
    # SummaryValidationData
    print('\n' + '='*70)
    print('FigureDataSummary (use paper )')
    print('='*70)
    print(f"Fig1 Internal AUC: {roc_data['internal_auc']:.4f}")
    print(f"Fig1 ExternalAUC: {roc_data['external_auc']:.4f}")
    print(f"Fig2 Brier:   {cal_data['brier']:.4f}")
    
    print('\nComplete! All FigurealreadyUpdate')


if __name__ == '__main__':
    main()
