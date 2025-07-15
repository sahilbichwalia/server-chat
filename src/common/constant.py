import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
DEFAULT_COLLECTION = os.getenv("MONGO_DEFAULT_COLLECTION")

BASE_PATH='../../pdf'
Data_PATH='data/fetch_latest_data.json'
CHROMA_DB_PATH = "./chroma_db"
# BASE_MODEL="gemma2:9b"
# BASE_MODEL="llama3.1:8b"
BASE_MODEL="gemini-2.0-flash"
EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
# MONGO_URI ="mongodb+srv://mehtaharsh2324:mehtaharsh123@mehtaharsh.nbvsczz.mongodb.net/"  # Or your connection string

JSON_DATA_PATH = 'data/fetch_latest_data.json'    