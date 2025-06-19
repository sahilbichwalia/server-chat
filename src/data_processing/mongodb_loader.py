# src/data_processing/mongodb_loader.py
from pymongo import MongoClient
from typing import List, Dict, Any
import logging
from src.common.constant import MONGO_URI, DB_NAME  # Add these to your constants
from src.common.constant import DEFAULT_COLLECTION
logger = logging.getLogger(__name__)

import ssl
from pymongo import MongoClient

class MongoDBLoader:
    def __init__(self, collection_name: str = None):
        if not collection_name:
            from src.common.constant import DEFAULT_COLLECTION
            collection_name = DEFAULT_COLLECTION
            
        if not isinstance(collection_name, str):
            raise ValueError(f"Collection name must be string, got {type(collection_name)}")
        
        self.client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            socketTimeoutMS=30000,
            connectTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        self.db = self.client[DB_NAME]
        self.collection = self.db[collection_name]
        
        # Verify connection
        try:
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB")
        except Exception as e:
            print(f"Connection failed: {e}")
            raise
    
    def load_all_servers(self) -> List[Dict[str, Any]]:
        """Load all server documents from MongoDB"""
        try:
            servers = list(self.collection.find({}))
            logger.info(f"Loaded {len(servers)} servers from MongoDB")
            return servers
        except Exception as e:
            logger.error(f"Failed to load from MongoDB: {str(e)}")
            raise
    
    def load_servers_by_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load servers with a specific query"""
        try:
            servers = list(self.collection.find(query))
            logger.info(f"Loaded {len(servers)} servers matching query")
            return servers
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()