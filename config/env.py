from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    RTSP_URL: str = Field(env="RTSP_URL", default="", description="URL video streaming")
    CAMERA_IP: str = Field(env="CAMERA_IP", default="", description="Camera's ip in local network")
    CAMERA_USERNAME: str = Field(env="CAMERA_USERNAME", default="", description="username to access camera")
    CAMERA_PASSWORD: str = Field(env="CAMERA_PASSWORD", default="", description="password to access camera")
    class Config:
        validate_assignment = True
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"  # Allow extra fields that are not defined in the model

settings = Settings()