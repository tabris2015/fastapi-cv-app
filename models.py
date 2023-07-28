from pydantic import BaseModel


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Prediction(BaseModel):
    bbox: BBox
    class_name: str
    confidence: float


class ObjectDetectionPrediction(BaseModel):
    file_name: str
    predictions: list[Prediction]
