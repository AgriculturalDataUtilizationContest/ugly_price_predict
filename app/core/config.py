from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    kamis_api_key: str = Field(..., alias="KAMIS_API_KEY")
    kamis_user_key: str = Field(..., alias="KAMIS_USER_KEY")

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True



settings = Settings()
