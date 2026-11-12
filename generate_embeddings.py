import os
import sys
from typing import Any, Dict, List
import uuid

from dotenv import load_dotenv
from documents_structure import structure_documents
from chunk_conversion import split_documents
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import numpy as np
import chromadb

load_dotenv()

class embeddingManager:


    def __init__(self,model_name:str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""

        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self,texts) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings



class VectorStore:
    def __init__(self, collection_name: str = "clinic_sops", persist_directory="./data/db/"):


        self.collection_name  = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()


    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: embeddingManager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Collection uses squared L2 distance; for unit-length embeddings
                    # (all-MiniLM-L6-v2 normalizes) d = 2 - 2*cos, so cos = 1 - d/2.
                    similarity_score = 1 - distance / 2
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


class GroqLLM:
    """Wraps a Groq chat model for answering questions from retrieved context."""

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an assistant that answers questions about clinic standard operating procedures (SOPs), policies and compliance guidance.
Answer using ONLY the context below. Cite the sources you used as [source, page N].
If the context does not contain enough information to answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}

Answer:""",
    )

    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env or pass api_key.")

        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model_name,
            temperature=0.1,
            max_tokens=1024,
        )
        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        prompt = self.PROMPT.format(context=context, question=query)
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"


def format_context(results: List[Dict[str, Any]]) -> str:
    """Turn retrieved results into a context block labelled with source file and page."""
    blocks = []
    for r in results:
        meta = r["metadata"]
        source = os.path.basename(str(meta.get("source", "unknown")))
        page = meta.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        blocks.append(f"[{source}{page_label}]\n{r['content']}")
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    embedding_manager = embeddingManager()
    vectorstore = VectorStore()
    retriever = RAGRetriever(embedding_manager=embedding_manager, vector_store=vectorstore)

    # Only load PDFs and embed them when the database is empty.
    if vectorstore.collection.count() == 0:
        docs = structure_documents()
        splitted_documents = split_documents(docs)
        texts = [doc.page_content for doc in docs]
        embeddings = embedding_manager.generate_embeddings(texts)
        vectorstore.add_documents(docs, embeddings=embeddings)

    # Ask a question: python generate_embeddings.py "your question"
    # or run with no arguments for an interactive prompt (blank line to quit).
    llm = GroqLLM()

    def ask(query: str, top_k: int = 5):
        results = retriever.retrieve(query, top_k=top_k)
        if not results:
            print("No relevant documents found for this question.")
            return

        answer = llm.generate_response(query, format_context(results))
        print("\nAnswer:\n" + answer)
        print("\nSources:")
        for r in results:
            meta = r["metadata"]
            page = meta["page"] + 1 if isinstance(meta.get("page"), int) else "?"
            print(f"  - {meta.get('source')} (page {page}), score {r['similarity_score']:.3f}")

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        while query := input("\nQuestion (blank to quit): ").strip():
            ask(query)


