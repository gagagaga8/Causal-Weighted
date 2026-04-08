"""
Run all ablation experiments
"""

import subprocess
import sys
import os

os.chdir('c:/Dynamic-RRT')

experiments = [
    ('ablation_timepoint.py', 'time pointAblation'),
    ('ablation_hyperparams.py', ' ParametersAblation'),
    ('ablation_data_ratio.py', 'Data Ablation'),
    ('ablation_class_weight.py', 'ClassWeightAblation'),
    ('ablation_feature_type.py', 'FeaturetypeAblation'),
]


def main():
    print("="*70)
    print("Run all ablation experiments")
    print("="*70)
    
    for script, name in experiments:
        print(f"\n{'='*70}")
        print(f"[{experiments.index((script, name))+1}/{len(experiments)}] {name}")
        print("="*70)
        
        result = subprocess.run(
            [sys.executable, f'Experiment /{script}'],
            capture_output=False
        )
        
        if result.returncode != 0:
            print(f"Warning: {script} ExecuteFailure")
    
    print("\n" + "="*70)
    print("All AblationExperimentComplete")
    print("="*70)
    print("\nGenerateResultsFile:")
    for f in os.listdir('Experiment '):
        if f.endswith('.csv'):
            print(f" - Experiment /{f}")


if __name__ == '__main__':
    main()
