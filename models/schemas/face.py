from pydantic import BaseModel, conlist
from typing import List, Any


class Face(BaseModel):
    data: List[conlist(float, min_items=4, max_items=4)]


class FacePredictionResponse(BaseModel):
    prediction: List[int]
    probability: List[Any]
    log_probability: List[Any]