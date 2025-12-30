from fastapi import FastAPI
from src.scripts.settings import Settings
from src.inference import SentimentAnalyzer

from api.prediction import PredictRequest, PredictResponse

settings = Settings()

app = FastAPI()

analyzer = SentimentAnalyzer(settings=settings)


@app.get("/")
def root():
    return {"message": "ONNX Sentiment App"}


@app.post("/predict", response_model=PredictResponse)
def inference(request: PredictRequest):
    response = analyzer.predict(request.text)
    return PredictResponse(sentiment=response)