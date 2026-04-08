import matplotlib.pyplot as plt
import numpy as np

# settingsall and 
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def generate_fairness_chart():
    subgroups = ['Age < 65', 'Age >= 65', 'Male', 'Female', 'SOFA < 5', 'SOFA 5-9', 'SOFA >= 10']
    auc_scores = [0.9225, 0.9245, 0.9227, 0.9241, 0.9287, 0.9180, 0.9285]
    colors = ['#4C72B0', '#4C72B0', '#55A868', '#55A868', '#DD8452', '#DD8452', '#DD8452']

    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_pos = np.arange(len(subgroups))
    bars = ax.barh(y_pos, auc_scores, color=colors, height=0.6)
    
    # settingsrange
    ax.set_xlim(0.85, 0.95)
    
    # Labelandtitle
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subgroups, fontsize=12)
    ax.set_xlabel('AUC-ROC Score', fontsize=14)
    ax.set_title('Algorithmic Fairness Assessment Across Clinical Subgroups', fontsize=16, pad=20)
    
    # in on 
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                ha='left', va='center', fontsize=12)
                
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # and 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('c:/Dynamic-RRT/figures/Fig13_Algorithmic_Fairness.pdf', format='pdf', dpi=300)
    plt.savefig('c:/Dynamic-RRT/figures/Fig13_Algorithmic_Fairness.png', format='png', dpi=300)
    print("Generated Fig13_Algorithmic_Fairness.pdf/.png")

if __name__ == "__main__":
    generate_fairness_chart()
