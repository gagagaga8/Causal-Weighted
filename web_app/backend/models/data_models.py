"""
DataModel 
 APIrequestandresponsePydanticModel
"""

from pydantic import BaseModel
from typing import Optional, Dict


class CaseFilter(BaseModel):
    """Case filter parameters"""
    dataset: Optional[str] = None  # "MIMIC" or "eICU"
    aki_stage: Optional[int] = None  # 1, 2, 3
    age_min: Optional[int] = None
    age_max: Optional[int] = None


class PredictRequest(BaseModel):
    """policyPredictionrequest"""
    case_id: Optional[str] = None  # Use existing cases
    custom_state: Optional[Dict[str, float]] = None # or Input
    timestep: int = 1  # 1, 2, or  3


class ExplainRequest(BaseModel):
    """explainabilityAnalysisrequest"""
    case_id: str
    timestep: int


class LLMExplainRequest(BaseModel):
    """LLM request"""
    case_id: str
    timestep: int
    language: str = "en"  # "en" or "zh"
