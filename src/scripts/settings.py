from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    s3_bucket_name: str = "aszarata-agh-mlops-lab11-model"
    s3_sentence_transformer_path: str = "sentence_transformer.model"
    s3_classifier_path: str = "classifier.joblib"
    
    base_dir: Path = Path(__file__).parent.parent.parent
    model_dir: Path = base_dir / "model"
    sentence_transformer_dir: Path = model_dir / "sentence_transformer.model"
    classifier_joblib_path: Path = model_dir / "classifier.joblib"

    onnx_embedding_model_path: Path = model_dir / "embedding_model.onnx"
    onnx_classifier_path: Path = model_dir / "classifier.onnx"
    onnx_tokenizer_path: Path = model_dir / "tokenizer.json"
    
    embedding_dim: int = 384