from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Project Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "datasets"
    RESULTS_DIR: Path = BASE_DIR / "results"

    # SROIE Dataset Config
    SROIE_FULL_PATH: Path = DATA_DIR / "sroie_full"
    SROIE_SAMPLES_PATH: Path = DATA_DIR / "sroie/samples"
    SROIE_SAMPLE_COUNT: int = 25

    # HF Dataset ID for fallback
    SROIE_HF_REPO: str = "priyank-m/SROIE_2019_text_recognition"

    # FUNSD Dataset Config
    FUNSD_FULL_PATH: Path = DATA_DIR / "funsd_full/dataset/testing_data"
    FUNSD_SAMPLES_PATH: Path = DATA_DIR / "funsd/samples"
    FUNSD_SAMPLE_COUNT: int = 25

    # External Service Config
    OCR_SPACE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
