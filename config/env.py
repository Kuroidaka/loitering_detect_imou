from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    RTSP_URL: str = Field(env="RTSP_URL", default="", description="URL video streaming")

    class Config:
        validate_assignment = True
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"  # Allow extra fields that are not defined in the model

settings = Settings()