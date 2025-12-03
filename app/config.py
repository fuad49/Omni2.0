from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # Facebook
    FACEBOOK_APP_SECRET: str
    FACEBOOK_VERIFY_TOKEN: str
    FACEBOOK_PAGE_ACCESS_TOKEN: str = "" # Optional default if needed

    # Security
    ENCRYPTION_KEY: str # Must be a valid Fernet key (32 url-safe base64-encoded bytes)

    # App
    ENV: str = "development"
    
    # Google Gemini
    GEMINI_API_KEY: str

    class Config:
        env_file = ".env"
        extra = "ignore" # Allow extra fields in .env
        case_sensitive = False # Allow case mismatch (e.g. supabase_url vs SUPABASE_URL)

@lru_cache()
def get_settings():
    return Settings()
