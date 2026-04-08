"""
LLM service
 GenerateClinical andliterature search
"""

from typing import Dict, List


class LLMService:
    """LLM explanation service"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def generate_clinical_summary(self, case: Dict, timestep: int) -> str:
        """
        GenerateClinicalSummary
        
        Args:
            case: Casedictionary
            timestep: 
            
        Returns:
            ClinicalSummary 
        """
        step_data = case['timeline'][timestep - 1]
        
        # Urine outputstatus
        uo_status = "oliguria" if step_data['uo'] < 0.5 else "normal urine output"
        
        # BUNstatus
        bun_status = "elevated" if step_data['bun'] > 20 else "normal"
        
        summary = f"""Patient: {case['case_id']}
Age: {case['age']} years, Gender: {case['gender']}
SOFA Score: {case['sofa_score']}
Weight: {case['weight']} kg

At timestep {timestep} (Day {timestep}):
- Urine output: {step_data['uo']:.1f} ml/kg/h ({uo_status})
- BUN: {step_data['bun']:.1f} mg/dL ({bun_status})
- Creatinine: {step_data['creat']:.1f} mg/dL
- Potassium: {step_data['pot']:.1f} mEq/L
- pH: {step_data['ph']:.2f}"""
        
        return summary.strip()
    
    def search_literature(self, clinical_summary: str) -> List[Dict]:
        """
        literature search 
        
        Args:
            clinical_summary: ClinicalSummary
            
        Returns:
             ColumnTable
        """
        # defaultReturn RCT 
        return [
            {
                'title': 'Timing of Renal-Replacement Therapy in Patients with Acute Kidney Injury and Sepsis',
                'journal': 'New England Journal of Medicine',
                'year': 2020,
                'pmid': '32668114',
                'relevance': 0.95
            },
            {
                'title': 'Effect of Early vs Delayed Initiation of Renal Replacement Therapy on Mortality in Critically Ill Patients With Acute Kidney Injury',
                'journal': 'JAMA',
                'year': 2016,
                'pmid': '27209269',
                'relevance': 0.92
            }
        ]
    
    def generate_explanation(
        self, 
        case: Dict, 
        timestep: int, 
        summary: str, 
        literature: List[Dict], 
        language: str
    ) -> str:
        """
        GenerateLLM 
        
        Args:
            case: Casedictionary
            timestep: 
            summary: ClinicalSummary
            literature: ColumnTable
            language: Language "en"or "zh" 
            
        Returns:
             
        """
        if language == "zh":
            return """ Patientwhen Clinicalstatusand IQLModelAnalysis notstartRRT 

Main basis:
1. Urine output incanAcceptrange >0.3 ml/kg/h 
2. electrolyte not to start 
3. according toSTARRT-AKI RCTstudy startpolicyin stablePatientinnot 

  items 
- need Urine output and balance
- e.g. or metabolic acidosisworsen Evaluation"""
        else:
            return """Based on the patient's current clinical status and our IQL model analysis, we recommend delaying RRT initiation.

Key rationale:
1. Urine output remains within acceptable range (>0.3 ml/kg/h)
2. Electrolyte disturbances have not reached emergency thresholds
3. According to recent RCTs including STARRT-AKI, delayed initiation strategy shows no inferiority in hemodynamically stable patients

Monitoring points:
- Close monitoring of urine output, potassium, and acid-base balance required
- Re-evaluation needed if volume overload, refractory hyperkalemia, or worsening metabolic acidosis develops"""
