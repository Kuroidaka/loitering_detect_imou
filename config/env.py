from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    RTSP_URL: str = Field(env="RTSP_URL", default="", description="URL video streaming")
    CAMERA_IP: str = Field(env="CAMERA_IP", default="", description="Camera's ip in local network")
    CAMERA_USERNAME: str = Field(env="CAMERA_USERNAME", default="", description="username to access camera")
    CAMERA_PASSWORD: str = Field(env="CAMERA_PASSWORD", default="", description="password to access camera")
    MIN_SPEED_TO_BE_LOITERING: float = Field(env="MIN_SPEED_TO_BE_LOITERING", default=0.5, description="Min speed to be paused")
    
    IMOU_APP_ID : str = Field(env="IMOU_APP_ID", default="", description="")
    IMOU_APP_SECRET : str = Field(env="IMOU_APP_SECRET", default="", description="")
    IMOU_CAMERA_ALIAS : str = Field(env="IMOU_CAMERA_ALIAS", default="", description="")
    
    TARGET_ZALO_GROUP_ID: str= Field(env="TARGET_ZALO_GROUP_ID", default="", description="")
    
    ZALO_NUMBER: str= Field(env="ZALO_NUMBER", default="", description="")
    ZALO_PASSWORD: str= Field(env="ZALO_PASSWORD", default="", description="")
    ZALO_IMEI: str= Field(env="ZALO_IMEI", default="", description="")
    ZALO_COOKIES: dict= Field(env="ZALO_COOKIES", default="", description="")

    class Config:
        validate_assignment = True
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"  # Allow extra fields that are not defined in the model

settings = Settings()