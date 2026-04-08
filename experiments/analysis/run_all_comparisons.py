"""
RunAll ComparisonExperiment
Execute all experiments and summarize results
"""

import os
import sys
import time
from datetime import datetime

# to items Directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_experiment(name, module_name):
    """Run Experiment"""
    print_header(f"Experiment: {name}")
    start_time = time.time()
    
    try:
        # and Run
        module = __import__(f'Experiment.{module_name}', fromlist=[module_name])
        
        # toand Run Function
        if hasattr(module, f'run_{module_name}'):
            getattr(module, f'run_{module_name}')()
        elif hasattr(module, 'main'):
            module.main()
        else:
            # RunModulein run_ Function
            for attr_name in dir(module):
                if attr_name.startswith('run_'):
                    getattr(module, attr_name)()
                    break
        
        elapsed = time.time() - start_time
        print(f"\n✓ {name} Complete (Time elapsed: {elapsed:.1f} seconds)")
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {name} Failure: {str(e)}")
        return False, elapsed


def run_all_experiments():
    """RunAll ComparisonExperiment"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*25 + "ComparisonExperiment Execute" + " "*25 + "║")
    print("╚" + "═"*78 + "╝")
    print(f"\nStart : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # All Experiment
    experiments = [
        ("SamplingpolicyComparison", "sampling_comparison"),
        ("Feature policyComparison", "feature_selection_comparison"),
        (" ", "timewindow_sensitivity"),
        ("set policyComparison", "ensemble_comparison"),
        ("Threshold Analysis", "threshold_analysis"),
    ]
    
    results = []
    total_start = time.time()
    
    for i, (name, module) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Run: {name}")
        print("-"*60)
        success, elapsed = run_experiment(name, module)
        results.append({
            'experiment': name,
            'success': success,
            'time': elapsed
        })
    
    total_elapsed = time.time() - total_start
    
    # Summaryreport
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*30 + "Executereport" + " "*30 + "║")
    print("╚" + "═"*78 + "╝")
    
    print(f"\n Time elapsed: {total_elapsed:.1f} seconds\n")
    print(f"{'Experiment ':30s} | {'status':8s} | {' ':10s}")
    print("-"*55)
    
    success_count = 0
    for r in results:
        status = "✓ Success" if r['success'] else "✗ Failure"
        print(f"{r['experiment']:30s} | {status:8s} | {r['time']:.1f} seconds")
        if r['success']:
            success_count += 1
    
    print("-"*55)
    print(f"\nSuccess: {success_count}/{len(experiments)}")
    
    # Column GenerateResultsFile
    print("\n" + "="*60)
    print("GenerateResultsFile:")
    print("="*60)
    
    result_files = [
        'Experiment/sampling_comparison.csv',
        'Experiment/feature_selection_comparison.csv',
        'Experiment/timewindow_sensitivity.csv',
        'Experiment/ensemble_comparison.csv',
        'Experiment/threshold_analysis.csv'
    ]
    
    for f in result_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✓ {f} ({size} bytes)")
        else:
            print(f"  ✗ {f} (notGenerate)")
    
    print(f"\nComplete : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == '__main__':
    run_all_experiments()
