"""
Supplementary experiment: Model calibration curves
Evaluate consistency between predicted probabilities and actual event rates
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]


def load_mimic_data():
    """LoadingMIMICData"""
    df = pd.read_csv('data/dwols_full_with_uo.csv')
    
    X = df[SAFE_FEATURES].copy()
    X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    X = X.fillna(0)
    
    valid_mask = ~df['hfd'].isna()
    X = X[valid_mask].values
    df_valid = df[valid_mask]
    T = ((df_valid['a1'] == 1) | (df_valid['a2'] == 1) | (df_valid['a3'] == 1)).astype(int).values
    
    return X, T


def load_eicu_data():
    """LoadingeICUData"""
    df = pd.read_csv('2_eICUPreprocessingdata/eicu_full_features.csv')
    df = df.drop_duplicates(subset=['patientunitstayid'])
    
    available = [c for c in SAFE_FEATURES if c in df.columns]
    X = df[available].copy()
    
    for f in SAFE_FEATURES:
        if f not in X.columns:
            X[f] = 0
    X = X[SAFE_FEATURES]
    
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0}).fillna(0)
    
    X = X.fillna(0).values
    T = df['received_rrt'].values
    
    return X, T


def get_cross_validated_probs(X, T, scaler):
    """ Cross-validationPredictionProbability"""
    X_scaled = scaler.fit_transform(X)
    
    all_probs = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, test_idx in skf.split(X_scaled, T):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        T_train = T[train_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X_train, T_train)
        all_probs[test_idx] = model.predict_proba(X_test)[:, 1]
    
    return all_probs


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """ComputationCalibration Metric"""
    # Brier Score
    brier = brier_score_loss(y_true, y_prob)
    
    # Calibration CurveData
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() * np.abs(bin_acc - bin_conf)
    ece /= len(y_prob)
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0
    
    return {
        'brier_score': brier,
        'ece': ece,
        'mce': mce,
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }


def plot_calibration_curves(mimic_metrics, eicu_metrics, save_path):
    """ Calibration Curve"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MIMICCalibration Curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(mimic_metrics['prob_pred'], mimic_metrics['prob_true'], 
            'o-', color='#3498DB', linewidth=2, markersize=8, label='LightGBM')
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
    ax.set_title('MIMIC-III Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Metric 
    textstr = f'Brier Score: {mimic_metrics["brier_score"]:.4f}\nECE: {mimic_metrics["ece"]:.4f}\nMCE: {mimic_metrics["mce"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # eICUCalibration Curve
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(eicu_metrics['prob_pred'], eicu_metrics['prob_true'], 
            'o-', color='#E74C3C', linewidth=2, markersize=8, label='LightGBM')
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
    ax.set_title('eICU External Validation Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    textstr = f'Brier Score: {eicu_metrics["brier_score"]:.4f}\nECE: {eicu_metrics["ece"]:.4f}\nMCE: {eicu_metrics["mce"]:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration CurvealreadySave: {save_path}")


def plot_probability_histogram(mimic_probs, mimic_labels, eicu_probs, eicu_labels, save_path):
    """ PredictionProbabilityDistribution """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MIMIC
    ax = axes[0]
    ax.hist(mimic_probs[mimic_labels == 0], bins=20, alpha=0.7, label='No RRT', color='#3498DB')
    ax.hist(mimic_probs[mimic_labels == 1], bins=20, alpha=0.7, label='RRT', color='#E74C3C')
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('MIMIC-III Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # eICU
    ax = axes[1]
    ax.hist(eicu_probs[eicu_labels == 0], bins=20, alpha=0.7, label='No RRT', color='#3498DB')
    ax.hist(eicu_probs[eicu_labels == 1], bins=20, alpha=0.7, label='RRT', color='#E74C3C')
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('eICU Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ProbabilityDistribution alreadySave: {save_path}")


def plot_reliability_diagram(y_true, y_prob, n_bins=10, save_path=None, title='Reliability Diagram'):
    """ can records """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_centers = []
    bin_accs = []
    bin_counts = []
    bin_stds = []
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() > 10:  # at least 10 samples
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_accs.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
            # Binomial standard error
            p = y_true[mask].mean()
            n = mask.sum()
            bin_stds.append(np.sqrt(p * (1-p) / n))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calibration 
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Calibration Curve records
    ax.errorbar(bin_centers, bin_accs, yerr=bin_stds, fmt='o-', 
                color='#3498DB', linewidth=2, markersize=8, capsize=5, 
                label='Model Calibration')
    
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"can alreadySave: {save_path}")
    else:
        plt.show()


def main():
    print("="*70)
    print("Supplementary experiment: Model calibration curves")
    print("="*70)
    
    # Load data
    print("\n[1/5] Load data")
    X_mimic, T_mimic = load_mimic_data()
    X_eicu, T_eicu = load_eicu_data()
    print(f"  MIMIC: {len(X_mimic)} samples, RRT Rate={T_mimic.mean():.1%}")
    print(f"  eICU: {len(X_eicu)} samples, RRT Rate={T_eicu.mean():.1%}")
    
    # Standardizeer
    scaler = StandardScaler()
    
    # MIMICCross-validationProbability
    print("\n[2/5] MIMICCross-validation")
    mimic_probs = get_cross_validated_probs(X_mimic, T_mimic, scaler)
    print(f"  PredictionProbabilityrange: [{mimic_probs.min():.3f}, {mimic_probs.max():.3f}]")
    
    # eICUPredictionProbability
    print("\n[3/5] eICUExternalValidation")
    scaler.fit(X_mimic)
    X_mimic_scaled = scaler.transform(X_mimic)
    X_eicu_scaled = scaler.transform(X_eicu)
    
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X_mimic_scaled, T_mimic)
    eicu_probs = model.predict_proba(X_eicu_scaled)[:, 1]
    print(f"  PredictionProbabilityrange: [{eicu_probs.min():.3f}, {eicu_probs.max():.3f}]")
    
    # ComputationCalibration Metric
    print("\n[4/5] ComputationCalibration Metric")
    mimic_metrics = compute_calibration_metrics(T_mimic, mimic_probs)
    eicu_metrics = compute_calibration_metrics(T_eicu, eicu_probs)
    
    print("\nMIMIC-IIICalibration Metric:")
    print(f"  Brier Score: {mimic_metrics['brier_score']:.4f}")
    print(f"  ECE (Expected Calibration Error): {mimic_metrics['ece']:.4f}")
    print(f"  MCE (Maximum Calibration Error): {mimic_metrics['mce']:.4f}")
    
    print("\neICUCalibration Metric:")
    print(f"  Brier Score: {eicu_metrics['brier_score']:.4f}")
    print(f"  ECE (Expected Calibration Error): {eicu_metrics['ece']:.4f}")
    print(f"  MCE (Maximum Calibration Error): {eicu_metrics['mce']:.4f}")
    
    # Figure
    print("\n[5/5] GenerateFigure")
    plot_calibration_curves(mimic_metrics, eicu_metrics, 'Visualization/figures/calibration_curves.png')
    plot_probability_histogram(mimic_probs, T_mimic, eicu_probs, T_eicu, 'Visualization/figures/probability_distribution.png')
    plot_reliability_diagram(T_mimic, mimic_probs, save_path='Visualization/figures/reliability_diagram_mimic.png', 
                            title='MIMIC-III Reliability Diagram')
    plot_reliability_diagram(T_eicu, eicu_probs, save_path='Visualization/figures/reliability_diagram_eicu.png',
                            title='eICU Reliability Diagram')
    
    # SaveResults
    results = pd.DataFrame([
        {'Dataset': 'MIMIC-III', 'Brier Score': mimic_metrics['brier_score'], 
         'ECE': mimic_metrics['ece'], 'MCE': mimic_metrics['mce']},
        {'Dataset': 'eICU', 'Brier Score': eicu_metrics['brier_score'], 
         'ECE': eicu_metrics['ece'], 'MCE': eicu_metrics['mce']}
    ])
    results.to_csv('Method /calibration_metrics.csv', index=False)
    print("\nResults saved: Method /calibration_metrics.csv")
    
    # Summary
    print("\n" + "="*70)
    print("Calibration EvaluationSummary")
    print("="*70)
    print(f"""
MIMIC-IIIInternal Validation 
- Brier Score: {mimic_metrics['brier_score']:.4f} ( 0 )
- ECE: {mimic_metrics['ece']:.4f} ( 0 )
- Calibration 

eICUExternalValidation 
- Brier Score: {eicu_metrics['brier_score']:.4f}
- ECE: {eicu_metrics['ece']:.4f}
- in Calibration bias paperinnoteDataDistribution 

Calibration Curve 
- Curve pair Table Calibration 
- Curveinpair on Table 
- Curveinpair down Table 
""")
    
    return results


if __name__ == '__main__':
    results = main()
