from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic.networks import AnyHttpUrl
from pathlib import Path

class Settings(BaseSettings):
    # Logging / Observability
    logfire_token: str = Field(..., env="LOGFIRE_TOKEN")
    environment: str     = Field("production", env="ENVIRONMENT")

    # Output directory
    output_dir: Path     = Field(Path("./outputs"), env="OUTPUT_DIR")

    # Risk-free Rate
    default_rf_rate: float = Field(0.05, env="DEFAULT_RF_RATE")

    # MOSEK License
    mosek_license_content: str | None = Field(None, env="MOSEK_LICENSE_CONTENT")
    mosek_license_path: Path  | None = Field(None, env="MOSEK_LICENSE_PATH")

    # CORS Origins
    allowed_origins: list[AnyHttpUrl] = Field(
        ["https://indportfoliooptimization.vercel.app"],
        env="ALLOWED_ORIGINS"
    )
    
    # FRED API Key for economic data
    fred_api_key: str | None = Field(None, env="FRED_API_KEY")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields in environment
    }

settings = Settings() 