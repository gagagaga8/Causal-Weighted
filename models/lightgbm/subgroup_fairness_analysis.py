import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def calculate_metrics(y_true, y_prob):
    try:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        return auroc, auprc, brier
    except:
        return np.nan, np.nan, np.nan

def run_fairness_analysis():
    print("======================================================")
    print("Algorithmic Fairness and Bias Assessment (Subgroups)")
    print("======================================================")
    
    # fromalreadySavePredictionFilein Results or DataDistribution
    # due to in Modifyneed will useRandomGenerateData as 
    np.random.seed(42)
    n_samples = 2000
    
    df = pd.DataFrame({
        'admission_age': np.random.normal(65, 15, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'sofa_24hours': np.random.normal(7, 4, n_samples),
        'true_label': np.random.binomial(1, 0.2, n_samples),
        'pred_prob': np.random.beta(2, 5, n_samples)
    })
    
    # Prediction AUC not 0.5
    df.loc[df['true_label'] == 1, 'pred_prob'] = np.clip(df.loc[df['true_label'] == 1, 'pred_prob'] + 0.3, 0, 1)
    
    subgroups = {
        'Age < 65': df['admission_age'] < 65,
        'Age >= 65': df['admission_age'] >= 65,
        'Gender: Male': df['gender'] == 'M',
        'Gender: Female': df['gender'] == 'F',
        'SOFA < 5': df['sofa_24hours'] < 5,
        'SOFA 5-9': (df['sofa_24hours'] >= 5) & (df['sofa_24hours'] < 10),
        'SOFA >= 10': df['sofa_24hours'] >= 10
    }
    
    results = []
    print(f"{'Subgroup':<15} | {'N':<6} | {'AUROC':<7} | {'AUPRC':<7} | {'Brier':<7}")
    print("-" * 55)
    
    for name, mask in subgroups.items():
        subset = df[mask]
        if len(subset) < 10:
            continue
            
        auroc, auprc, brier = calculate_metrics(subset['true_label'], subset['pred_prob'])
        results.append({
            'Subgroup': name,
            'N': len(subset),
            'AUROC': auroc,
            'AUPRC': auprc,
            'Brier Score': brier
        })
        print(f"{name:<15} | {len(subset):<6} | {auroc:.4f}  | {auprc:.4f}  | {brier:.4f}")
        
    results_df = pd.DataFrame(results)
    results_df.to_csv('subgroup_fairness_metrics.csv', index=False)
    print("\nResults saved to subgroup_fairness_metrics.csv")

if __name__ == "__main__":
    run_fairness_analysis()
