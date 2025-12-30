from src.scripts.settings import Settings


def test_settings_s3_bucket():
    settings = Settings()
    assert settings.s3_bucket == "mlops-lab11-models-bhanc"


def test_settings_embedding_dim():
    settings = Settings()
    assert settings.embedding_dim == 384