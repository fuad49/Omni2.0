import os
import sys

# Add current directory to path so we can import app modules
sys.path.append(os.getcwd())

try:
    from app.ai_engine import load_models
    print("Preloading models...")
    load_models()
    print("Models preloaded successfully.")
except ImportError:
    # If app.ai_engine doesn't exist yet (during initial build steps if copy order is different), 
    # we might need to handle this. But with 'COPY . .' it should be there.
    # However, for the very first run before ai_engine is created, this might fail if run locally.
    # In Docker, we copy everything.
    # To be safe, we will create a dummy ai_engine if it doesn't exist for the sake of this script 
    # but strictly speaking, this script depends on ai_engine.py being present.
    print("Could not import app.ai_engine. Ensure the file exists.")
    sys.exit(1)
except Exception as e:
    print(f"Error preloading models: {e}")
    sys.exit(1)
