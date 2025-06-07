from src.common.constant import BASE_PATH, CHROMA_DB_PATH
from src.common.common import LLM_INSTANCE, Embeddings
from langchain.memory import ConversationBufferMemory
import os
import glob
from typing import List, Optional
from src.config.logging_config import setup_logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from src.config.logging_config import setup_logging
logger = setup_logging()

def load_pdf_documents(base_path: str = BASE_PATH) -> List:
    """Load PDF documents from the specified base path"""
    logger.info(f"Loading PDF documents from {base_path}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(base_path):
        logger.warning(f"Document path {base_path} does not exist. Creating directory.")
        os.makedirs(base_path, exist_ok=True)
        return []
    
    entries = glob.glob(os.path.join(base_path, "*"))
    documents = []

    for entry in entries:
        if os.path.isdir(entry):
            pdf_files = glob.glob(os.path.join(entry, "**", "*.pdf"), recursive=True)
        elif entry.lower().endswith(".pdf"):
            pdf_files = [entry]
        else:
            continue

        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                
                # Add better metadata handling
                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(pdf_file)
                    doc.metadata["file_path"] = pdf_file
                    doc.metadata["file_size"] = os.path.getsize(pdf_file)
                    documents.append(doc)
                    
                logger.info(f"Loaded PDF: {pdf_file} ({len(docs)} pages)")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                continue  # Continue with other files instead of breaking
    
    logger.info(f"Loaded {len(documents)} document chunks from {len(set(doc.metadata['source_file'] for doc in documents))} files")
    return documents

def setup_vector_store(documents: List) -> Optional[object]:
    """Set up vector store with the provided documents"""
    if not documents:
        logger.warning("No documents provided to setup_vector_store")
        return None
        
    logger.info("Setting up vector store")
    
    # Ensure chroma_db directory exists
    chroma_db_path = CHROMA_DB_PATH
    if not os.path.exists(chroma_db_path):
        logger.info(f"Creating ChromaDB directory: {chroma_db_path}")
        os.makedirs(chroma_db_path, exist_ok=True)
    
    # Use RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]  # Better separation strategy
    )
    
    try:
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(split_docs)} chunks")
        
        # Initialize embeddings with error handling
        embeddings = Embeddings
        
        # Create vector store with better error handling
        try:
            # Check if existing ChromaDB exists and try to load it first
            if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
                try:
                    logger.info("Attempting to load existing ChromaDB...")
                    vectorstore = Chroma(
                        persist_directory=chroma_db_path,
                        embedding_function=embeddings
                    )
                    # Test if the vectorstore works
                    test_results = vectorstore.similarity_search("test", k=1)
                    logger.info("Successfully loaded existing ChromaDB")
                    return vectorstore
                except Exception as load_error:
                    logger.warning(f"Could not load existing ChromaDB: {load_error}. Creating new one.")
            
            # Create new vector store
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=chroma_db_path
            )
            vectorstore.persist()  # Explicitly persist
            logger.info("Persistent vector store created successfully")
            
        except Exception as persist_error:
            logger.warning(f"Error creating persistent vector store: {persist_error}")
            logger.info("Falling back to in-memory vector store")
            vectorstore = Chroma.from_documents(
                documents=split_docs, 
                embedding=embeddings
            )
            logger.info("In-memory vector store created")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error in setup_vector_store: {e}")
        return None

def setup_rag_chain(vectorstore: object, llm_instance: object) -> Optional[object]:
    """Set up and return the RAG conversation chain with memory"""
    if not vectorstore:
        logger.error("No vectorstore provided to setup_rag_chain")
        return None
        
    if not llm_instance:
        logger.error("No LLM instance provided to setup_rag_chain")
        return None
        
    try:
        # Set up the retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # Explicitly set search type
            search_kwargs={"k": 3}
        )
        
        # Set up conversation buffer memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Create the conversational retrieval chain with memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=LLM_INSTANCE,
            retriever=retriever,
            chain_type="stuff",
            memory=memory,
            return_source_documents=True,
            verbose=False,
            max_tokens_limit=4000  # Add token limit to prevent context overflow
        )
        
        logger.info("RAG chain with conversation memory created successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {e}")
        return None

# -----------------------------
# RAG Tool Functions
# -----------------------------
# Global RAG variables
rag_chain: Optional[object] = None
vectorstore: Optional[object] = None

def query_documents(query: str) -> str:
    """Query the document knowledge base using RAG"""
    global rag_chain
    
    if not rag_chain:
        return "Document search is not available. No documents have been loaded. Please initialize the RAG system first."
    
    if not query or not query.strip():
        return "Please provide a specific question to search the documents."
    
    try:
        # Add chat history parameter (empty for stateless queries)
        response = rag_chain({
            "question": query.strip(),
            "chat_history": []  # Required parameter for ConversationalRetrievalChain
        })
        
        answer = response.get("answer", "No answer found.")
        source_docs = response.get("source_documents", [])
        
        formatted_response = f"**Answer:** {answer}\n\n"
        
        if source_docs:
            formatted_response += "**Sources:**\n"
            seen_sources = set()
            for doc in source_docs[:3]:  # Limit to top 3 sources
                source_file = doc.metadata.get("source_file", "Unknown")
                if source_file not in seen_sources:
                    formatted_response += f"- {source_file}\n"
                    seen_sources.add(source_file)
        else:
            formatted_response += "**Sources:** No relevant sources found.\n"
        
        return formatted_response.strip()
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return f"Error occurred while searching documents: {str(e)}"

def list_available_documents() -> str:
    """List all available documents in the knowledge base"""
    global vectorstore
    
    if not vectorstore:
        return "No documents are currently loaded in the knowledge base. Please initialize the RAG system first."
    
    try:
        all_docs = vectorstore.get()
        
        if not all_docs or not all_docs.get('metadatas'):
            return "No documents found in the knowledge base."
        
        source_files = set()
        total_chunks = 0
        
        for metadata in all_docs['metadatas']:
            if metadata and isinstance(metadata, dict):  # Add type check
                source_file = metadata.get('source_file', 'Unknown')
                source_files.add(source_file)
                total_chunks += 1
        
        if not source_files:
            return "No valid documents found in the knowledge base."
        
        response = f"**Available Documents ({len(source_files)} files):**\n"
        for source_file in sorted(source_files):
            response += f"- {source_file}\n"
        
        response += f"\n**Total document chunks:** {total_chunks}"
        return response
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return "Error occurred while listing available documents."

# -----------------------------
# Initialize RAG System
# -----------------------------
def initialize_rag_system(llm_instance: object, documents_path: str = "C:\\Users\\sahil\\Desktop\\chat\\pdf") -> bool:
    """
    Initialize RAG system - call this in your setup_agent function
    Returns True if successful, False otherwise
    """
    global rag_chain, vectorstore
    
    logger.info("Initializing RAG system...")
    
    if not llm_instance:
        logger.error("No LLM instance provided to initialize_rag_system")
        return False
    
    try:
        # Check if we already have a vectorstore loaded
        if vectorstore and rag_chain:
            logger.info("RAG system already initialized")
            return True
            
        documents = load_pdf_documents(documents_path)
        
        if not documents:
            logger.warning("No documents found - RAG system will be unavailable")
            return False
            
        vectorstore = setup_vector_store(documents)
        if not vectorstore:
            logger.error("Vector store setup failed")
            return False
            
        rag_chain = setup_rag_chain(vectorstore, llm_instance)
        if not rag_chain:
            logger.error("RAG chain setup failed")
            return False
            
        logger.info("RAG system initialized successfully")
        return True
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        rag_chain = None
        vectorstore = None
        return False

def auto_initialize_rag_on_startup(llm_instance: object) -> bool:
    """
    Automatically initialize RAG system on startup
    This function will be called during system initialization
    """
    logger.info("Starting automatic RAG system initialization...")
    
    # List of possible document directories to check
    possible_paths = [
        "./pdf",
         BASE_PATH
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found document directory: {path}")
            success = initialize_rag_system(llm_instance, path)
            if success:
                return True
        else:
            logger.info(f"Document directory not found: {path}")
    
    # If no existing directories found, create default one
    default_path = "./documents"
    logger.info(f"Creating default documents directory: {default_path}")
    os.makedirs(default_path, exist_ok=True)
    
    # Try to initialize with empty directory (will result in no documents but system ready)
    success = initialize_rag_system(llm_instance, default_path)
    
    if success:
        logger.info("RAG system initialized with empty document directory")
    else:
        logger.warning("RAG system initialization failed even with empty directory")
    
    return success

# -----------------------------
# Utility Functions
# -----------------------------
def reset_rag_system() -> None:
    """Reset the RAG system (useful for reloading documents)"""
    global rag_chain, vectorstore
    rag_chain = None
    vectorstore = None
    logger.info("RAG system reset")

def get_rag_system_status() -> dict:
    """Get the current status of the RAG system"""
    global rag_chain, vectorstore
    
    return {
        "rag_chain_initialized": rag_chain is not None,
        "vectorstore_initialized": vectorstore is not None,
        "system_ready": rag_chain is not None and vectorstore is not None,
        "chroma_db_exists": os.path.exists("./chroma_db"),
        "documents_directory_exists": os.path.exists("./documents")
    }

def ensure_directories_exist():
    """Ensure all required directories exist"""
    directories = [BASE_PATH,CHROMA_DB_PATH]
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")
            