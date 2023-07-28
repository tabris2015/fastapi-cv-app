from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from models import ObjectDetectionPrediction
from predictor import ObjectDetectorPredictor

app = FastAPI(title="Vision Inference Service")     # crea una nueva app de fastapi

od_predictor = ObjectDetectorPredictor()


@app.post("/predict_object_detection")
def predict_object_detection(file: UploadFile = File(...)) -> ObjectDetectionPrediction:
    is_valid_image = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not is_valid_image:
        raise HTTPException(
            status_code=415,
            detail="Archivo no soportado"
        )
    # imagen valida, realizamos la prediccion
    result = od_predictor.predict_file(file)
    return result


@app.get("/")
def root():
    return {"status": "OK"}


# cuando se ejecuta el archivo como un script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
