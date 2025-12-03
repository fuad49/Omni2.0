from cryptography.fernet import Fernet
from app.config import get_settings

settings = get_settings()
cipher_suite = Fernet(settings.ENCRYPTION_KEY)

def encrypt_token(token: str) -> str:
    """Encrypts a token using Fernet symmetric encryption."""
    if not token:
        return ""
    return cipher_suite.encrypt(token.encode()).decode()

def decrypt_token(token: str) -> str:
    """Decrypts a token using Fernet symmetric encryption."""
    if not token:
        return ""
    return cipher_suite.decrypt(token.encode()).decode()
