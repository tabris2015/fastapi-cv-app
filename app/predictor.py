import io
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.models import BBox, Prediction, ObjectDetectionPrediction

MODEL_PATH = "nn_models/ssd_mobilenet_v2.tflite"


class ObjectDetectorPredictor:
    def __init__(self) -> None:
        self.base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        self.options = vision.ObjectDetectorOptions(
            base_options=self.base_options,
            score_threshold=0.5
        )
        self.detector = vision.ObjectDetector.create_from_options(self.options)

    def predict(self, img: np.ndarray, filename: str) -> ObjectDetectionPrediction:
        """Realiza una prediccion sobre una imagen de tipo np.ndarray"""
        input_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img
        )
        detections = self.detector.detect(input_image)
        return self.process_detections(filename, detections)

    def predict_file(self, file) -> ObjectDetectionPrediction:
        """Realiza una prediccion sobre un archivo de tipo file"""
        img_stream = io.BytesIO(file.file.read())
        img_stream.seek(0)
        img = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        return self.predict(img, file.filename)

    def process_detections(self, filename, detections) -> ObjectDetectionPrediction:
        """Procesa los datos de las predicciones de mediapipe a los modelos definidos"""
        predictions = []
        for detection in detections.detections:
            bbox = BBox(
                x=detection.bounding_box.origin_x,
                y=detection.bounding_box.origin_y,
                w=detection.bounding_box.width,
                h=detection.bounding_box.height,
            )
            prediction = Prediction(
                bbox=bbox,
                class_name=detection.categories[0].category_name,
                confidence=detection.categories[0].score,
            )
            predictions.append(prediction)

        result = ObjectDetectionPrediction(
            file_name=filename,
            predictions=predictions
        )

        return result
