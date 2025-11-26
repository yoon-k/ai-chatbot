import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
import hashlib
import math
import re
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
import numpy as np

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError

logger = structlog.get_logger()


# ============================================================
# BM25 Implementation
# ============================================================
class BM25:
    """BM25 (Best Matching 25) sparse retrieval algorithm."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0-1, typically 0.75)
            epsilon: Floor value for IDF
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.tokenized_docs: List[List[str]] = []

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization with basic preprocessing."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stopwords (minimal set)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'own', 'same', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'any', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'whose'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def fit(self, documents: List[str]) -> 'BM25':
        """
        Fit BM25 on a corpus of documents.

        Args:
            documents: List of document texts

        Returns:
            Self for chaining
        """
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.corpus_size = len(documents)
        self.doc_len = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in self.tokenized_docs:
            seen = set()
            for word in doc:
                if word not in seen:
                    self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
                    seen.add(word)

        # Calculate IDF
        self.idf = {}
        for word, freq in self.doc_freqs.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[word] = max(idf, self.epsilon)

        return self

    def get_scores(self, query: str) -> np.ndarray:
        """
        Calculate BM25 scores for all documents given a query.

        Args:
            query: Query string

        Returns:
            Array of BM25 scores for each document
        """
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size)

        for i, doc in enumerate(self.tokenized_docs):
            doc_len = self.doc_len[i]
            term_freqs = Counter(doc)

            for term in query_tokens:
                if term in term_freqs:
                    freq = term_freqs[term]
                    idf = self.idf.get(term, self.epsilon)

                    # BM25 formula
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[i] += idf * numerator / denominator

        return scores

    def get_top_k(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top-k documents for a query.

        Args:
            query: Query string
            k: Number of results

        Returns:
            List of (doc_index, score) tuples
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ============================================================
# Query Expansion
# ============================================================
class QueryExpander:
    """Query expansion using various techniques."""

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def expand_with_llm(
        self,
        query: str,
        num_expansions: int = 3
    ) -> List[str]:
        """
        Expand query using LLM to generate related queries.

        Args:
            query: Original query
            num_expansions: Number of expanded queries to generate

        Returns:
            List of expanded queries including original
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"""Generate {num_expansions} alternative search queries
                        that capture different aspects or phrasings of the user's question.
                        Return only the queries, one per line, no numbering."""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=200
            )

            expanded = response.choices[0].message.content.strip().split('\n')
            expanded = [q.strip() for q in expanded if q.strip()]

            return [query] + expanded[:num_expansions]

        except Exception as e:
            logger.warning("Query expansion failed", error=str(e))
            return [query]

    def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Expand query with common synonyms.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        # Simple synonym mapping
        synonyms = {
            'how': ['what way', 'what method'],
            'find': ['locate', 'search', 'get'],
            'create': ['make', 'build', 'generate'],
            'delete': ['remove', 'erase'],
            'update': ['modify', 'change', 'edit'],
            'problem': ['issue', 'error', 'bug'],
            'fix': ['solve', 'resolve', 'repair'],
            'best': ['optimal', 'recommended', 'top'],
            'fast': ['quick', 'rapid', 'efficient'],
        }

        expanded = [query]
        query_lower = query.lower()

        for word, syns in synonyms.items():
            if word in query_lower:
                for syn in syns[:2]:  # Max 2 synonyms per word
                    expanded.append(query_lower.replace(word, syn))

        return expanded[:5]  # Max 5 total


# ============================================================
# Document Chunking Strategies
# ============================================================
class SmartChunker:
    """Advanced document chunking with multiple strategies."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Different splitters for different content types
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

    def detect_content_type(self, text: str) -> str:
        """Detect the type of content."""
        # Check for code patterns
        code_patterns = [
            r'def \w+\(',
            r'class \w+:',
            r'function \w+\(',
            r'import \w+',
            r'from \w+ import',
            r'const \w+ =',
            r'let \w+ =',
            r'var \w+ =',
        ]

        for pattern in code_patterns:
            if re.search(pattern, text):
                return 'code'

        # Check for markdown
        if re.search(r'^#{1,6}\s', text, re.MULTILINE):
            return 'markdown'

        # Check for table-like structure
        if text.count('|') > 5:
            return 'table'

        return 'prose'

    def chunk_text(self, text: str, content_type: str = None) -> List[Dict[str, Any]]:
        """
        Chunk text based on content type.

        Args:
            text: Text to chunk
            content_type: Optional content type override

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if content_type is None:
            content_type = self.detect_content_type(text)

        if content_type == 'code':
            return self._chunk_code(text)
        elif content_type == 'markdown':
            return self._chunk_markdown(text)
        else:
            return self._chunk_prose(text)

    def _chunk_prose(self, text: str) -> List[Dict[str, Any]]:
        """Standard prose chunking."""
        chunks = self.recursive_splitter.split_text(text)
        return [
            {
                "text": chunk,
                "type": "prose",
                "index": i,
                "char_count": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

    def _chunk_code(self, text: str) -> List[Dict[str, Any]]:
        """Code-aware chunking preserving function/class boundaries."""
        chunks = []
        current_chunk = []
        current_size = 0

        # Split by function/class definitions
        lines = text.split('\n')
        function_start_patterns = [
            r'^def \w+',
            r'^class \w+',
            r'^async def \w+',
            r'^function \w+',
            r'^const \w+ = \(',
            r'^export ',
        ]

        for line in lines:
            is_boundary = any(re.match(p, line) for p in function_start_patterns)

            if is_boundary and current_size > self.chunk_size // 2:
                # Start new chunk at boundary
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line) + 1

                if current_size >= self.chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return [
            {
                "text": chunk,
                "type": "code",
                "index": i,
                "char_count": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

    def _chunk_markdown(self, text: str) -> List[Dict[str, Any]]:
        """Markdown-aware chunking preserving headers."""
        chunks = []
        current_header = ""
        current_content = []
        current_size = 0

        for line in text.split('\n'):
            # Check for header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save current chunk if exists
                if current_content:
                    chunk_text = f"{current_header}\n" + '\n'.join(current_content)
                    chunks.append({
                        "text": chunk_text.strip(),
                        "type": "markdown",
                        "header": current_header,
                        "index": len(chunks)
                    })

                current_header = line
                current_content = []
                current_size = 0
            else:
                current_content.append(line)
                current_size += len(line) + 1

                if current_size >= self.chunk_size:
                    chunk_text = f"{current_header}\n" + '\n'.join(current_content)
                    chunks.append({
                        "text": chunk_text.strip(),
                        "type": "markdown",
                        "header": current_header,
                        "index": len(chunks)
                    })
                    current_content = []
                    current_size = 0

        # Don't forget last chunk
        if current_content:
            chunk_text = f"{current_header}\n" + '\n'.join(current_content)
            chunks.append({
                "text": chunk_text.strip(),
                "type": "markdown",
                "header": current_header,
                "index": len(chunks)
            })

        # Add char_count to all chunks
        for chunk in chunks:
            chunk["char_count"] = len(chunk["text"])

        return chunks


# ============================================================
# Reranking
# ============================================================
class Reranker:
    """Rerank retrieved documents for better relevance."""

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def rerank_with_llm(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using LLM.

        Args:
            query: Original query
            documents: Retrieved documents
            top_k: Number of documents to return

        Returns:
            Reranked documents
        """
        if len(documents) <= top_k:
            return documents

        # Create prompt for reranking
        doc_texts = []
        for i, doc in enumerate(documents[:20]):  # Limit to 20 for efficiency
            content = doc.get("content", doc.get("text", ""))[:500]
            doc_texts.append(f"[{i}] {content}")

        docs_str = "\n\n".join(doc_texts)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a document relevance ranker. Given a query and documents,
                        return the indices of the most relevant documents in order of relevance.
                        Return only the indices as comma-separated numbers, e.g.: 3,0,5,2,1"""
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nDocuments:\n{docs_str}\n\nReturn top {top_k} most relevant document indices:"
                    }
                ],
                temperature=0,
                max_tokens=50
            )

            # Parse indices
            indices_str = response.choices[0].message.content.strip()
            indices = [int(i.strip()) for i in indices_str.split(',') if i.strip().isdigit()]

            # Reorder documents
            reranked = []
            for idx in indices[:top_k]:
                if idx < len(documents):
                    doc = documents[idx].copy()
                    doc["rerank_position"] = len(reranked)
                    reranked.append(doc)

            return reranked

        except Exception as e:
            logger.warning("LLM reranking failed", error=str(e))
            return documents[:top_k]

    def rerank_with_cross_encoder_score(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Simple cross-encoder style reranking using term overlap.

        Args:
            query: Original query
            documents: Retrieved documents
            top_k: Number of documents to return

        Returns:
            Reranked documents
        """
        query_terms = set(query.lower().split())

        scored_docs = []
        for doc in documents:
            content = doc.get("content", doc.get("text", "")).lower()
            doc_terms = set(content.split())

            # Calculate overlap score
            overlap = len(query_terms & doc_terms)
            total = len(query_terms)
            overlap_score = overlap / total if total > 0 else 0

            # Combine with original score
            original_score = doc.get("similarity_score", doc.get("final_score", 0.5))
            combined_score = 0.7 * original_score + 0.3 * overlap_score

            scored_doc = doc.copy()
            scored_doc["rerank_score"] = combined_score
            scored_docs.append(scored_doc)

        # Sort by combined score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_k]


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

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        use_query_expansion: bool = True,
        use_reranking: bool = True
    ):
        self.rag = rag_pipeline
        self.bm25_index: Dict[str, BM25] = {}  # user_id -> BM25 index
        self.doc_store: Dict[str, List[Dict]] = {}  # user_id -> documents
        self.use_query_expansion = use_query_expansion
        self.use_reranking = use_reranking
        self.query_expander = QueryExpander(rag_pipeline.openai_client)
        self.reranker = Reranker(rag_pipeline.openai_client)
        self.smart_chunker = SmartChunker()

    def build_bm25_index(
        self,
        documents: List[Dict[str, Any]],
        user_id: str
    ) -> None:
        """
        Build BM25 index for a user's documents.

        Args:
            documents: List of documents with 'content' field
            user_id: User ID for index storage
        """
        texts = [doc.get("content", doc.get("text", "")) for doc in documents]
        self.bm25_index[user_id] = BM25().fit(texts)
        self.doc_store[user_id] = documents
        logger.info("BM25 index built", user_id=user_id, doc_count=len(documents))

    async def retrieve(
        self,
        query: str,
        document_ids: List[str],
        user_id: str,
        top_k: int = 10,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        expand_query: bool = None,
        rerank: bool = None
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval using both dense and sparse methods.

        Args:
            query: Search query
            document_ids: Documents to search
            user_id: User ID
            top_k: Number of results
            dense_weight: Weight for dense (embedding) retrieval
            sparse_weight: Weight for sparse (BM25) retrieval
            expand_query: Override for query expansion
            rerank: Override for reranking

        Returns:
            Combined and reranked results
        """
        if expand_query is None:
            expand_query = self.use_query_expansion
        if rerank is None:
            rerank = self.use_reranking

        # Optionally expand query
        queries = [query]
        if expand_query:
            try:
                queries = await self.query_expander.expand_with_llm(query, num_expansions=2)
            except Exception as e:
                logger.warning("Query expansion failed", error=str(e))

        # Collect results from all queries
        all_dense_results = []
        all_sparse_results = []

        for q in queries:
            # Dense retrieval
            dense_results = await self.rag.retrieve(
                query=q,
                document_ids=document_ids,
                user_id=user_id,
                top_k=top_k * 2
            )
            all_dense_results.extend(dense_results)

            # Sparse (BM25) retrieval
            if user_id in self.bm25_index:
                bm25 = self.bm25_index[user_id]
                docs = self.doc_store[user_id]

                # Filter by document_ids if provided
                if document_ids:
                    filtered_indices = [
                        i for i, doc in enumerate(docs)
                        if doc.get("document_id") in document_ids
                    ]
                else:
                    filtered_indices = list(range(len(docs)))

                if filtered_indices:
                    # Get BM25 scores
                    bm25_results = bm25.get_top_k(q, k=top_k * 2)

                    for idx, score in bm25_results:
                        if idx in filtered_indices:
                            doc = docs[idx]
                            all_sparse_results.append({
                                "content": doc.get("content", doc.get("text", "")),
                                "metadata": doc.get("metadata", {}),
                                "bm25_score": score,
                                "document_id": doc.get("document_id"),
                                "chunk_index": doc.get("chunk_index", idx)
                            })

        # Reciprocal Rank Fusion (RRF)
        result_scores = {}
        k_rrf = 60  # RRF constant

        # Process dense results
        for i, result in enumerate(all_dense_results):
            doc_id = f"{result.get('document_id', '')}_{result.get('chunk_index', i)}"
            rank = i + 1
            rrf_score = dense_weight / (k_rrf + rank)

            if doc_id in result_scores:
                result_scores[doc_id]["rrf_score"] += rrf_score
            else:
                result_scores[doc_id] = {
                    **result,
                    "rrf_score": rrf_score,
                    "sources": ["dense"]
                }

        # Process sparse results
        for i, result in enumerate(all_sparse_results):
            doc_id = f"{result.get('document_id', '')}_{result.get('chunk_index', i)}"
            rank = i + 1
            rrf_score = sparse_weight / (k_rrf + rank)

            if doc_id in result_scores:
                result_scores[doc_id]["rrf_score"] += rrf_score
                result_scores[doc_id]["sources"].append("sparse")
                result_scores[doc_id]["bm25_score"] = result.get("bm25_score", 0)
            else:
                result_scores[doc_id] = {
                    **result,
                    "rrf_score": rrf_score,
                    "sources": ["sparse"]
                }

        # Sort by RRF score
        sorted_results = sorted(
            result_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # Apply final score
        for result in sorted_results:
            result["final_score"] = result["rrf_score"]

        # Optionally rerank
        if rerank and len(sorted_results) > top_k:
            try:
                sorted_results = await self.reranker.rerank_with_llm(
                    query=query,
                    documents=sorted_results[:top_k * 2],
                    top_k=top_k
                )
            except Exception as e:
                logger.warning("Reranking failed", error=str(e))
                sorted_results = sorted_results[:top_k]
        else:
            sorted_results = sorted_results[:top_k]

        return sorted_results

    async def retrieve_with_context(
        self,
        query: str,
        document_ids: List[str],
        user_id: str,
        top_k: int = 5,
        context_window: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with surrounding context chunks.

        Args:
            query: Search query
            document_ids: Documents to search
            user_id: User ID
            top_k: Number of results
            context_window: Number of chunks before/after to include

        Returns:
            Results with surrounding context
        """
        results = await self.retrieve(
            query=query,
            document_ids=document_ids,
            user_id=user_id,
            top_k=top_k
        )

        if user_id not in self.doc_store:
            return results

        docs = self.doc_store[user_id]

        # Enhance results with context
        enhanced_results = []
        for result in results:
            chunk_idx = result.get("chunk_index", 0)
            doc_id = result.get("document_id")

            # Find surrounding chunks from same document
            context_before = []
            context_after = []

            for i, doc in enumerate(docs):
                if doc.get("document_id") == doc_id:
                    doc_chunk_idx = doc.get("chunk_index", i)
                    if chunk_idx - context_window <= doc_chunk_idx < chunk_idx:
                        context_before.append(doc.get("content", ""))
                    elif chunk_idx < doc_chunk_idx <= chunk_idx + context_window:
                        context_after.append(doc.get("content", ""))

            enhanced = result.copy()
            enhanced["context_before"] = context_before
            enhanced["context_after"] = context_after
            enhanced["full_context"] = "\n\n".join(
                context_before + [result.get("content", "")] + context_after
            )
            enhanced_results.append(enhanced)

        return enhanced_results


class AdvancedRAGPipeline(RAGPipeline):
    """Enhanced RAG Pipeline with advanced features."""

    def __init__(self):
        super().__init__()
        self.smart_chunker = SmartChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.hybrid_retriever = HybridRetriever(self)
        self.query_expander = QueryExpander(self.openai_client)
        self.reranker = Reranker(self.openai_client)

    async def process_document_smart(
        self,
        file_path: str,
        file_type: str,
        document_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document with smart chunking based on content type.

        Args:
            file_path: Path to the document
            file_type: File type
            document_id: Document ID
            user_id: User ID
            metadata: Additional metadata

        Returns:
            Processing results
        """
        try:
            # Load document
            text = await self._load_document(file_path, file_type)

            # Smart chunking
            chunks = self.smart_chunker.chunk_text(text)

            # Generate embeddings
            chunk_texts = [c["text"] for c in chunks]
            embeddings = await self._generate_embeddings(chunk_texts)

            # Store in ChromaDB
            collection_name = f"user_{user_id}"
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # Prepare data
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_type": chunk.get("type", "prose"),
                    "user_id": user_id,
                    **(metadata or {})
                }
                for i, chunk in enumerate(chunks)
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )

            # Build BM25 index for hybrid retrieval
            doc_entries = [
                {
                    "content": chunk["text"],
                    "document_id": document_id,
                    "chunk_index": i,
                    "metadata": metadatas[i]
                }
                for i, chunk in enumerate(chunks)
            ]

            # Get or create user's document store
            if user_id in self.hybrid_retriever.doc_store:
                self.hybrid_retriever.doc_store[user_id].extend(doc_entries)
            else:
                self.hybrid_retriever.doc_store[user_id] = doc_entries

            # Rebuild BM25 index
            self.hybrid_retriever.build_bm25_index(
                self.hybrid_retriever.doc_store[user_id],
                user_id
            )

            logger.info(
                "Document processed with smart chunking",
                document_id=document_id,
                chunk_count=len(chunks),
                content_types=list(set(c.get("type") for c in chunks))
            )

            return {
                "document_id": document_id,
                "chunk_count": len(chunks),
                "total_characters": len(text),
                "collection_name": collection_name,
                "chunk_types": {
                    ctype: sum(1 for c in chunks if c.get("type") == ctype)
                    for ctype in set(c.get("type") for c in chunks)
                }
            }

        except Exception as e:
            logger.error("Smart document processing failed", error=str(e))
            raise DocumentProcessingError(file_path, str(e))

    async def hybrid_retrieve(
        self,
        query: str,
        document_ids: List[str],
        user_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid (dense + sparse) method.

        Args:
            query: Search query
            document_ids: Documents to search
            user_id: User ID
            top_k: Number of results

        Returns:
            Retrieved and reranked results
        """
        return await self.hybrid_retriever.retrieve(
            query=query,
            document_ids=document_ids,
            user_id=user_id,
            top_k=top_k
        )

    async def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate answer from retrieved context.

        Args:
            query: User's question
            context_docs: Retrieved documents
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer with sources
        """
        # Build context string
        context_parts = []
        for i, doc in enumerate(context_docs):
            content = doc.get("content", doc.get("text", ""))
            source = doc.get("metadata", {}).get("source", f"Document {i+1}")
            context_parts.append(f"[Source: {source}]\n{content}")

        context_str = "\n\n---\n\n".join(context_parts)

        default_system = """You are a helpful assistant that answers questions based on the provided context.
        Always cite your sources using [Source: ...] format.
        If the context doesn't contain enough information, say so clearly.
        Be concise but thorough."""

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt or default_system},
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nQuestion: {query}"
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": [
                    {
                        "document_id": doc.get("document_id"),
                        "content_preview": doc.get("content", "")[:200],
                        "score": doc.get("final_score", doc.get("similarity_score", 0))
                    }
                    for doc in context_docs
                ],
                "model": settings.OPENAI_MODEL,
                "context_used": len(context_docs)
            }

        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return {
                "answer": "I encountered an error generating the answer. Please try again.",
                "error": str(e),
                "sources": []
            }
