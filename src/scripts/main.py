import os
from src.scripts.download_artifacts import download_from_s3
from src.scripts.export_classifier_to_onnx import export_classifier_to_onnx
from src.scripts.export_sentence_transformer_to_onnx import export_model_to_onnx
from src.scripts.settings import Settings

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    settings = Settings()
    print("Downloading model")
    download_from_s3(settings)
    print("Classifier to onnx")
    export_classifier_to_onnx(settings)
    print("Model to onnx")
    export_model_to_onnx(settings)