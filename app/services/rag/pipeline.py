import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError

logger = structlog.get_logger()


class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) pipeline for document Q&A."""

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    async def process_document(
        self,
        file_path: str,
        file_type: str,
        document_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document and store embeddings.

        Args:
            file_path: Path to the document file
            file_type: Type of document (pdf, docx, txt, etc.)
            document_id: Unique document identifier
            user_id: Owner user ID
            metadata: Additional metadata

        Returns:
            Processing result with chunk count and stats
        """
        try:
            # Load document
            logger.info("Loading document", document_id=document_id, file_type=file_type)
            text = await self._load_document(file_path, file_type)

            # Split into chunks
            logger.info("Splitting document", document_id=document_id)
            chunks = self.text_splitter.split_text(text)

            # Generate embeddings
            logger.info("Generating embeddings", document_id=document_id, chunk_count=len(chunks))
            embeddings = await self._generate_embeddings(chunks)

            # Store in ChromaDB
            collection_name = f"user_{user_id}"
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # Prepare data for ChromaDB
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "user_id": user_id,
                    **(metadata or {})
                }
                for i in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )

            logger.info(
                "Document processed successfully",
                document_id=document_id,
                chunk_count=len(chunks)
            )

            return {
                "document_id": document_id,
                "chunk_count": len(chunks),
                "total_characters": len(text),
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error("Document processing failed", document_id=document_id, error=str(e))
            raise DocumentProcessingError(file_path, str(e))

    async def retrieve(
        self,
        query: str,
        document_ids: List[str],
        user_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Search query
            document_ids: List of document IDs to search in
            user_id: User ID for collection selection
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])

            # Search in ChromaDB
            if user_id:
                collection_name = f"user_{user_id}"
            else:
                # Search across all collections (admin mode)
                collection_name = "global"

            try:
                collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                logger.warning("Collection not found", collection_name=collection_name)
                return []

            # Build filter for document IDs
            where_filter = {
                "document_id": {"$in": document_ids}
            } if document_ids else None

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity (cosine)
                similarity = 1 - distance

                if similarity >= similarity_threshold:
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": similarity,
                        "document_id": metadata.get("document_id"),
                        "chunk_index": metadata.get("chunk_index")
                    })

            return formatted_results

        except Exception as e:
            logger.error("Retrieval failed", error=str(e))
            return []

    async def delete_document(
        self,
        document_id: str,
        user_id: str
    ) -> bool:
        """Delete all chunks for a document.

        Args:
            document_id: Document to delete
            user_id: Owner user ID

        Returns:
            True if deleted successfully
        """
        try:
            collection_name = f"user_{user_id}"
            collection = self.chroma_client.get_collection(collection_name)

            # Delete by document_id filter
            collection.delete(
                where={"document_id": document_id}
            )

            logger.info("Document deleted from vector store", document_id=document_id)
            return True

        except Exception as e:
            logger.error("Document deletion failed", document_id=document_id, error=str(e))
            return False

    async def _load_document(self, file_path: str, file_type: str) -> str:
        """Load document content based on file type."""

        loader_map = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "txt": TextLoader,
            "md": TextLoader,
            "html": UnstructuredHTMLLoader
        }

        loader_class = loader_map.get(file_type.lower())
        if not loader_class:
            raise DocumentProcessingError(file_path, f"Unsupported file type: {file_type}")

        loader = loader_class(file_path)
        documents = loader.load()

        # Combine all pages/sections
        text = "\n\n".join(doc.page_content for doc in documents)
        return text

    async def _generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI."""

        response = await self.openai_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts
        )

        return [item.embedding for item in response.data]


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods."""

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline

    async def retrieve(
        self,
        query: str,
        document_ids: List[str],
        user_id: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval using both dense and sparse methods.

        Args:
            query: Search query
            document_ids: Documents to search
            user_id: User ID
            top_k: Number of results
            dense_weight: Weight for dense (embedding) retrieval
            sparse_weight: Weight for sparse (BM25) retrieval

        Returns:
            Combined and reranked results
        """
        # Dense retrieval
        dense_results = await self.rag.retrieve(
            query=query,
            document_ids=document_ids,
            user_id=user_id,
            top_k=top_k * 2  # Get more for reranking
        )

        # TODO: Add BM25 sparse retrieval
        # For now, just use dense results

        # Score fusion (Reciprocal Rank Fusion)
        result_scores = {}
        for i, result in enumerate(dense_results):
            doc_id = f"{result['document_id']}_{result['chunk_index']}"
            rank = i + 1
            rrf_score = dense_weight / (60 + rank)
            result_scores[doc_id] = {
                **result,
                "final_score": rrf_score
            }

        # Sort by final score and return top_k
        sorted_results = sorted(
            result_scores.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]

        return sorted_results
