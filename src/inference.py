from src.scripts.settings import Settings
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort

SENTIMENT_MAP = {0: "negative", 1: "positive"}

class SentimentAnalyzer:
    def __init__(self, settings: Settings):
        self.tokenizer = Tokenizer.from_file(str(settings.onnx_tokenizer_path))
        
        self.embedding_session = ort.InferenceSession(str(settings.onnx_embedding_model_path), providers=['CPUExecutionProvider'])
        self.classifier_session = ort.InferenceSession(str(settings.onnx_classifier_path), providers=['CPUExecutionProvider'])

    def predict(self, text: str):
        # tokenize input
        encoded = self.tokenizer.encode(text)

        # prepare numpy arrays for ONNX
        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])

        # run embedding inference
        embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        embeddings = self.embedding_session.run(None, embedding_inputs)[0]

        # run classifier inference
        classifier_input_name = self.classifier_session.get_inputs()[0].name
        classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
        prediction = self.classifier_session.run(None, classifier_inputs)[0]

        predicted_class = int(np.argmax(prediction))

        return SENTIMENT_MAP.get(predicted_class, "unknown")