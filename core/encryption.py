"""Encryption for API keys and passwords"""
from cryptography.fernet import Fernet
import os
from pathlib import Path

class EncryptionManager:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)

    def _get_or_create_key(self) -> bytes:
        # Try environment variable first
        key_str = os.getenv("ENCRYPTION_KEY")
        if key_str:
            return key_str.encode()

        # Try key file
        key_file = Path(__file__).parent.parent / "data" / ".encryption_key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()

        # Generate new key
        key = Fernet.generate_key()
        key_file.parent.mkdir(exist_ok=True)
        with open(key_file, "wb") as f:
            f.write(key)
        os.chmod(key_file, 0o600)

        print(f"⚠️  Generated encryption key: {key_file}")
        print("⚠️  BACKUP THIS FILE - You cannot decrypt data without it!")
        return key

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        return self.cipher.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext:
            return ""
        try:
            return self.cipher.decrypt(ciphertext.encode()).decode()
        except Exception as e:
            print(f"❌ Decryption failed: {e}")
            return ""

encryption = EncryptionManager()
