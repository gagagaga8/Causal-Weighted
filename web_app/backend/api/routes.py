"""
API 
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from models.data_models import PredictRequest, ExplainRequest, LLMExplainRequest
import os

# Select data loader based on environment variable
USE_DEMO_MODE = os.getenv("USE_DEMO_MODE", "true").lower() == "true"

if USE_DEMO_MODE:
    from services.demo_data_loader import demo_data_loader as data_loader
else:
    from services.data_loader import data_loader
from services.prediction_service import PredictionService
from services.explainability_service import ExplainabilityService
from services.llm_service import LLMService
from services.pubmed_service import get_pubmed_service

# Create router
router = APIRouter(prefix="/api")

# Create service instance
prediction_service = PredictionService(data_loader)
explainability_service = ExplainabilityService(data_loader)
llm_service = LLMService(data_loader)


@router.get("/cases")
async def get_cases(
    dataset: Optional[str] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    limit: int = 20
):
    """
     CaseColumnTable
    
    Parameters:
        dataset: DatasetFilter ("MIMIC" or  "eICU")
        age_min: Minimumage
        age_max: Maximumage
        limit: Return 
    """
    if data_loader.case_database is None:
        raise HTTPException(status_code=500, detail="CaseData notLoading")
    
    # FilterCase
    filtered = data_loader.case_database.copy()
    
    if dataset:
        filtered = [c for c in filtered if c['dataset'] == dataset]
    
    if age_min:
        filtered = [c for c in filtered if c['age'] >= age_min]
    
    if age_max:
        filtered = [c for c in filtered if c['age'] <= age_max]
    
    # limit 
    filtered = filtered[:limit]
    
    return {
        'total': len(filtered),
        'cases': filtered
    }


@router.get("/cases/{case_id}")
async def get_case_detail(case_id: str):
    """ Case Info"""
    if data_loader.case_database is None:
        raise HTTPException(status_code=500, detail="CaseData notLoading")
    
    case = next((c for c in data_loader.case_database if c['case_id'] == case_id), None)
    
    if not case:
        raise HTTPException(status_code=404, detail="Casenot to")
    
    return case


@router.post("/predict")
async def predict_action(request: PredictRequest):
    """
    GenerateLightGBMpolicy and ComparisondWOLSandPhysician 
    
    Return:
        - LightGBM (Probability )
        - dWOLS (crude/stringent)
        - Physician 
    """
    # status
    if request.case_id:
        # fromCaseID 
        case = next(
            (c for c in data_loader.case_database if c['case_id'] == request.case_id), 
            None
        )
        
        if not case:
            raise HTTPException(status_code=404, detail="Casenot to")
        
        try:
            state = prediction_service.get_state_from_case(case, request.timestep)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        actual_action = case['timeline'][request.timestep - 1]['actual_action']
        original_pid = case['original_pid']
        
    elif request.custom_state:
        # Input
        state = request.custom_state
        actual_action = None
        original_pid = None
    else:
        raise HTTPException(status_code=400, detail="needto case_idor custom_state")
    
    # LightGBMPrediction demo mode use Prediction 
    try:
        if USE_DEMO_MODE and hasattr(data_loader, 'predict'):
            # demo mode - use Prediction
            demo_pred = data_loader.predict(state)
            lgb_prediction = {
                'action': demo_pred['recommended_action'],
                'action_name': 'Start RRT' if demo_pred['recommended_action'] == 1 else 'Do Not Start',
                'probability': {
                    'no_start': 1.0 - demo_pred['rrt_probability'],
                    'start': demo_pred['rrt_probability']
                },
                'confidence': demo_pred['confidence'],
                'risk_level': demo_pred['risk_level']
            }
        else:
            # - useLightGBMModel
            lgb_prediction = prediction_service.predict_lightgbm(state)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # dWOLS 
    dwols_recommendation = None
    if original_pid is not None:
        dwols_recommendation = prediction_service.get_dwols_recommendation(
            original_pid, 
            request.timestep
        )
    
    return {
        'state': state,
        'timestep': request.timestep,
        'predictions': {
            'lightgbm': lgb_prediction,
            'dwols': dwols_recommendation,
            'actual': {
                'action': actual_action,
                'action_name': (
                    'Start RRT' if actual_action == 1 
                    else 'Do Not Start' if actual_action == 0 
                    else 'Unknown'
                )
            }
        }
    }


@router.post("/explain")
async def explain_decision(request: ExplainRequest):
    """
    GenerateSHAP and Analysis
    
    Return:
        - SHAPvalue use waterfall 
        - CurveData
        - Qvalue Curve
    """
    # Case
    case = next(
        (c for c in data_loader.case_database if c['case_id'] == request.case_id), 
        None
    )
    
    if not case:
        raise HTTPException(status_code=404, detail="Casenot to")
    
    # status
    try:
        base_state = prediction_service.get_state_from_case(case, request.timestep)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # SHAPAnalysis
    try:
        shap_values = explainability_service.compute_shap_values(base_state)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Analysis
    try:
        counterfactual_results = explainability_service.perform_counterfactual_analysis(
            base_state
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        'case_id': request.case_id,
        'timestep': request.timestep,
        'base_state': base_state,
        'shap': shap_values,
        'counterfactual': counterfactual_results
    }


@router.post("/llm_explain")
async def llm_explain(request: LLMExplainRequest):
    """
     useLLMGenerateClinical literature search 
    """
    # Case
    case = next(
        (c for c in data_loader.case_database if c['case_id'] == request.case_id), 
        None
    )
    
    if not case:
        raise HTTPException(status_code=404, detail="Casenot to")
    
    # GenerateClinicalSummary
    clinical_summary = llm_service.generate_clinical_summary(case, request.timestep)
    
    # literature search
    relevant_literature = llm_service.search_literature(clinical_summary)
    
    # LLMGenerate 
    llm_explanation = llm_service.generate_explanation(
        case, 
        request.timestep, 
        clinical_summary, 
        relevant_literature, 
        request.language
    )
    
    return {
        'case_id': request.case_id,
        'timestep': request.timestep,
        'clinical_summary': clinical_summary,
        'literature': relevant_literature,
        'explanation': llm_explanation
    }


@router.get("/health")
async def health_check():
    """ """
    return {
        "status": "healthy",
        "model_loaded": data_loader.lgb_model is not None,
        "data_loaded": data_loader.dwols_policies is not None,
        "cases_loaded": data_loader.case_database is not None
    }


# ============================================================================
# PubMedliterature searchAPI
# ============================================================================

@router.get("/pubmed/search")
async def pubmed_search(query: str, max_results: int = 10):
    """
    searchPubMed 
    
    Parameters:
        query: search 
        max_results: ReturnMaximumResults default10 
    """
    pubmed = get_pubmed_service()
    papers = pubmed.search(query, max_results)
    return {
        'query': query,
        'total': len(papers),
        'papers': papers
    }


@router.get("/pubmed/rrt-related")
async def pubmed_rrt_related(max_results: int = 20):
    """
    searchRRTCorrelation Query 
    """
    pubmed = get_pubmed_service()
    papers = pubmed.search_rrt_related(max_results)
    return {
        'query': 'RRT + AKI + ML/RL',
        'total': len(papers),
        'papers': papers
    }


@router.post("/pubmed/search-keywords")
async def pubmed_search_keywords(keywords: list, max_results: int = 10):
    """
    according to ColumnTablesearch
    
    Parameters:
        keywords: ColumnTable
        max_results: ReturnMaximumResults 
    """
    pubmed = get_pubmed_service()
    papers = pubmed.search_by_keywords(keywords, max_results)
    return {
        'keywords': keywords,
        'total': len(papers),
        'papers': papers
    }
