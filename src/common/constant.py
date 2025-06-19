BASE_PATH='../../pdf'
Data_PATH='data/fetch_latest_data.json'
CHROMA_DB_PATH = "./chroma_db"
# BASE_MODEL="gemma2:9b"
# BASE_MODEL="llama3.1:8b"
BASE_MODEL="gemini-2.0-flash"
EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
# MONGO_URI ="mongodb+srv://mehtaharsh2324:mehtaharsh123@mehtaharsh.nbvsczz.mongodb.net/"  # Or your connection string
MONGO_URI = "mongodb+srv://mehtaharsh2324:mehtaharsh123@cluster0.nbvsczz.mongodb.net/<dbname>?retryWrites=true&w=majority&tls=true"
DB_NAME = "server_data"
DEFAULT_COLLECTION = "servers"
JSON_DATA_PATH = 'data/fetch_latest_data.json'    