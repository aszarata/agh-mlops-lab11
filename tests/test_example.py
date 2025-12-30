import sys
sys.path.append('src')
from scripts.settings import Settings


def test_settings_s3_bucket():
    settings = Settings()
    assert settings.s3_bucket_name == "aszarata-agh-mlops-lab11-model"


def test_settings_embedding_dim():
    settings = Settings()
    assert settings.embedding_dim == 384