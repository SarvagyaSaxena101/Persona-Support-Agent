import os
import pickle
import logging
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Handles document embedding, FAISS indexing, and semantic search using LangChain.
    Uses free local HuggingFace embeddings (all-MiniLM-L6-v2).
    """

    def __init__(
        self, 
        kb_dir: str = "data/raw", 
        index_path: str = "data/index"
    ):
        self.kb_dir = kb_dir
        self.index_path = index_path
        
        # Load local free embeddings (runs on your CPU)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store: Optional[FAISS] = None

        # Ensure index directory exists
        os.makedirs(self.index_path, exist_ok=True)

    def build_index(self):
        """
        Loads .txt files, splits them into chunks, and builds a FAISS index.
        """
        if not os.path.exists(self.kb_dir):
            logger.warning(f"KB directory {self.kb_dir} not found.")
            return

        # Load all .txt files from the directory
        loader = DirectoryLoader(self.kb_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            logger.warning("No documents found in KB directory.")
            return

        # Split documents into chunks with overlap for better context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)

        # Build FAISS index
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

        # Save index locally
        self.vector_store.save_local(self.index_path)
        logger.info(f"FAISS index built and saved to {self.index_path}")

    def load_index(self):
        """
        Loads the FAISS index from disk.
        """
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully.")
        else:
            logger.warning("No existing index found. Building a new one...")
            self.build_index()

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Performs semantic search to find relevant context for a given query.
        """
        if not self.vector_store:
            self.load_index()

        if not self.vector_store:
            return []

        # Search for similar documents
        results = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
