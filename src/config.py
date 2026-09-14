from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")