"""
ChromaDB client for semantic search functionality.

This module provides persistent vector database storage and embedding functions
for semantic search over Zotero libraries.
"""

import json
import os
import gc
from pathlib import Path
from typing import Any
import logging

try:
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        "chromadb is required for semantic search. "
        "Install it with: pip install 'zotero-mcp-server[semantic]'"
    ) from e

from zotero_mcp.utils import suppress_stdout

logger = logging.getLogger(__name__)


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Custom OpenAI embedding function for ChromaDB."""

    max_input_tokens = 8000  # text-embedding-3-* limit is 8191

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None, base_url: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            import openai
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")

    @staticmethod
    def name() -> str:
        return "openai"

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "base_url": self.base_url}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OpenAIEmbeddingFunction":
        return OpenAIEmbeddingFunction(
            model_name=config.get("model_name", "text-embedding-3-small"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string. No special handling needed for OpenAI."""
        return self.__call__([text])[0]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using tiktoken cl100k_base (correct for OpenAI models)."""
        try:
            import tiktoken
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            tokens = self._tokenizer.encode(text, disallowed_special=())
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = self._tokenizer.decode(tokens)
        except ImportError:
            max_chars = max_tokens * 3
            if len(text) > max_chars:
                text = text[:max_chars]
        return text


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom Gemini embedding function for ChromaDB using google-genai."""

    # gemini-embedding-2-* models ignore the task_type config field (the API
    # silently drops it). Google's recommended alternative is to embed the
    # task instruction in the prompt text itself, which empirically shifts
    # the embedding space (cos ~0.84 vs raw baseline) and preserves asymmetric
    # doc/query tuning (cos ~0.94 between doc-prefix and query-prefix).
    # These are the canonical prefixes; __call__ and embed_query prepend them
    # to every v2 input. They MUST stay in sync with V2_PREFIX_TOKEN_BUDGET
    # below: if you lengthen a prefix, bump the budget so truncation still
    # leaves room for it under the model's hard cap.
    V2_DOC_PREFIX = "Represent this document for retrieval:\n\n"
    V2_QUERY_PREFIX = "Represent this query for retrieval:\n\n"

    # Token reservation for the v2 prefix above. The longest prefix is
    # V2_DOC_PREFIX at 42 chars ~= 11 tokens with typical English tokenization.
    # We reserve 20 tokens (11 actual + 9 slack) so that truncate() leaves
    # room for the prefix without ever producing a post-prefix payload that
    # exceeds the model's 8192 hard cap even on dense text.
    V2_PREFIX_TOKEN_BUDGET = 20

    # Default for gemini-embedding-001 (hard cap 2048 tokens). Per-instance
    # override in __init__ for models with larger context windows. NOTE: for
    # v2 models this value means "effective budget for the TEXT BODY" —
    # prefix tokens are reserved separately (see V2_PREFIX_TOKEN_BUDGET).
    max_input_tokens = 2000

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: str | None = None, base_url: str | None = None):
        self.model_name = model_name
        # Model-aware token limit. For v2 models, derive from:
        #   hard_cap (8192) - safety_margin (192, for char-based truncation
        #   imprecision) - V2_PREFIX_TOKEN_BUDGET (20, reserved for the
        #   in-prompt task instruction prepended in __call__/embed_query).
        # Net effective budget for text body: 8192 - 192 - 20 = 7980 tokens.
        # This guarantees post-prefix payload <= hard cap even at the
        # truncation limit, formally closing the cap-enforcement gap.
        if "gemini-embedding-2" in model_name:
            self.max_input_tokens = 8000 - self.V2_PREFIX_TOKEN_BUDGET
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
        if not self.api_key:
            raise ValueError("Gemini API key is required")

        try:
            from google import genai
            from google.genai import types
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                http_options = types.HttpOptions(baseUrl=self.base_url)
                client_kwargs["http_options"] = http_options
            self.client = genai.Client(**client_kwargs)
            self.types = types
        except ImportError:
            raise ImportError("google-genai package is required for Gemini embeddings")

    @staticmethod
    def name() -> str:
        return "gemini"

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "base_url": self.base_url}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(
            model_name=config.get("model_name", "gemini-embedding-001"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
        )

    # Gemini's embed_content API caps at 100 items per batch (verified
    # empirically: batch=100 OK, batch=250 → 400 INVALID_ARGUMENT with
    # "at most 100 requests can be in one batch").
    GEMINI_MAX_BATCH = 100

    def _is_v2(self) -> bool:
        # gemini-embedding-2-* does not support the task_type config field
        # (it is silently ignored by the API). Google's guidance is to put
        # the task hint in the prompt text instead.
        return "gemini-embedding-2" in self.model_name

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using Gemini API, batching up to 100 per call."""
        is_v2 = self._is_v2()
        # Materialize once so we can slice regardless of input iterable type.
        texts = list(input)
        if is_v2:
            # v2 models: task instruction goes in the prompt, no config.
            # V2_PREFIX_TOKEN_BUDGET is already reserved from max_input_tokens
            # in __init__, so upstream truncation guarantees the combined
            # payload stays under the model's hard cap.
            prepared = [f"{self.V2_DOC_PREFIX}{t}" for t in texts]
        else:
            prepared = texts

        embeddings: list = []
        for start in range(0, len(prepared), self.GEMINI_MAX_BATCH):
            batch = prepared[start:start + self.GEMINI_MAX_BATCH]
            if is_v2:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                )
            else:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=self.types.EmbedContentConfig(
                        task_type="retrieval_document",
                        title="Zotero library document",
                    ),
                )
            embeddings.extend(e.values for e in response.embeddings)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string using retrieval_query task type."""
        # Truncate before any prefix prepending. For v2 models max_input_tokens
        # already excludes V2_PREFIX_TOKEN_BUDGET (reserved in __init__), so
        # the post-prefix payload stays under the model's hard cap. For v1
        # models truncation prevents API errors on pathological queries that
        # the upstream pipeline does not pre-truncate (queries bypass the
        # _process_item_batch truncate_text path that documents go through).
        text = self.truncate(text, self.max_input_tokens)
        if self._is_v2():
            prompt_text = f"{self.V2_QUERY_PREFIX}{text}"
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[prompt_text],
            )
        else:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_query",
                ),
            )
        return response.embeddings[0].values

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using character-based estimation for Gemini (~4 chars/token)."""
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        return text


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str | None = None,
        batch_size: int | None = None,
        torch_dtype: str | None = None,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.device:
                kwargs["device"] = self.device
            if self.torch_dtype:
                kwargs["model_kwargs"] = {
                    "dtype": self._resolve_torch_dtype(self.torch_dtype)
                }
            self.model = SentenceTransformer(model_name, **kwargs)
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

        # Read limit from model metadata; conservative fallback
        self.max_input_tokens = getattr(self.model, "max_seq_length", 500)

    def _resolve_device(self, device: str | None) -> str | None:
        """Resolve optional device config while preserving default behavior."""
        if not device:
            return None

        normalized = str(device).strip().lower()
        if normalized == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

        return normalized

    def _resolve_torch_dtype(self, torch_dtype: str) -> Any:
        """Resolve configured torch dtype for HuggingFace model loading."""
        normalized = str(torch_dtype).strip().lower()
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is required when embedding_config.torch_dtype is set") from e

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in dtype_map:
            raise ValueError(
                "Unsupported embedding_config.torch_dtype "
                f"'{torch_dtype}'. Use auto, float16, bfloat16, or float32."
            )
        return dtype_map[normalized]

    def _cleanup_device_cache(self) -> None:
        """Release cached CUDA/ROCm memory after embedding calls."""
        if not self.device or not self.device.startswith("cuda"):
            return
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Unable to clear CUDA cache after embedding", exc_info=True)

    @staticmethod
    def name() -> str:
        return "huggingface"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "torch_dtype": self.torch_dtype,
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HuggingFaceEmbeddingFunction":
        return HuggingFaceEmbeddingFunction(
            model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
            device=config.get("device"),
            batch_size=config.get("batch_size"),
            torch_dtype=config.get("torch_dtype"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        encode_kwargs: dict[str, Any] = {"convert_to_numpy": True}
        if self.batch_size:
            encode_kwargs["batch_size"] = int(self.batch_size)
        try:
            embeddings = self.model.encode(input, **encode_kwargs)
            return embeddings.tolist()
        finally:
            self._cleanup_device_cache()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string. No special handling needed for HuggingFace."""
        return self.__call__([text])[0]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using the model's own tokenizer."""
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is not None:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > max_tokens:
                encoded = encoded[:max_tokens]
                text = tokenizer.decode(encoded)
        else:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                text = text[:max_chars]
        return text


class ChromaClient:
    """ChromaDB client for Zotero semantic search."""

    def __init__(self,
                 collection_name: str = "zotero_library",
                 persist_directory: str | None = None,
                 embedding_model: str = "default",
                 embedding_config: dict[str, Any] | None = None):
        """
        Initialize ChromaDB client.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ('default', 'openai', 'gemini', 'qwen', 'embeddinggemma', or HuggingFace model name)
            embedding_config: Configuration for the embedding model
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}

        # Set up persistent directory
        if persist_directory is None:
            # Use user's config directory by default
            config_dir = Path.home() / ".config" / "zotero-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            persist_directory = str(config_dir / "chroma_db")

        self.persist_directory = persist_directory

        # Initialize ChromaDB client with stdout suppression
        with suppress_stdout():
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Set up embedding function
            self.embedding_function = self._create_embedding_function()

            # Get or create collection with the configured embedding function.
            # If the user switched embedding models, the persisted collection
            # will have stale config.  Detect the mismatch and drop/recreate.
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )

                # ChromaDB may silently persist the old embedding function config.
                # Check if the stored config matches what we want; if not, recreate.
                stored_config = getattr(self.collection, 'metadata', {}) or {}
                if not stored_config:
                    # Try reading config from the collection's config_json_str
                    try:
                        import json as _json
                        rows = self.client._sysdb.get_collections(name=self.collection_name)
                        if rows:
                            raw = getattr(rows[0], 'config_json_str', None) or '{}'
                            cfg = _json.loads(raw)
                            ef_cfg = cfg.get('embedding_function', {}).get('config', {})
                            stored_model = ef_cfg.get('model_name', '')
                            # Compare stored model with configured model
                            configured_model = getattr(self.embedding_function, 'model_name', None)
                            if stored_model and configured_model and stored_model != configured_model:
                                logger.warning(
                                    f"Stored embedding model '{stored_model}' differs from "
                                    f"configured '{configured_model}'. Resetting collection."
                                )
                                self.client.delete_collection(name=self.collection_name)
                                self.collection = self.client.create_collection(
                                    name=self.collection_name,
                                    embedding_function=self.embedding_function
                                )
                    except Exception:
                        pass  # Best-effort check; proceed with existing collection

            except Exception as e:
                if "embedding function conflict" in str(e).lower():
                    logger.warning(
                        f"Embedding model changed to '{self.embedding_model}'. "
                        "Resetting collection for rebuild."
                    )
                    self.client.delete_collection(name=self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function
                    )
                else:
                    raise

    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create the appropriate embedding function based on configuration."""
        if self.embedding_model == "openai":
            model_name = self.embedding_config.get("model_name", "text-embedding-3-small")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key, base_url=base_url)

        elif self.embedding_model == "gemini":
            model_name = self.embedding_config.get("model_name", "gemini-embedding-001")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return GeminiEmbeddingFunction(model_name=model_name, api_key=api_key, base_url=base_url)

        elif self.embedding_model == "qwen":
            model_name = self.embedding_config.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
            return HuggingFaceEmbeddingFunction(
                model_name=model_name,
                device=self.embedding_config.get("device"),
                batch_size=self.embedding_config.get("batch_size"),
                torch_dtype=self.embedding_config.get("torch_dtype"),
            )

        elif self.embedding_model == "embeddinggemma":
            model_name = self.embedding_config.get("model_name", "google/embeddinggemma-300m")
            return HuggingFaceEmbeddingFunction(
                model_name=model_name,
                device=self.embedding_config.get("device"),
                batch_size=self.embedding_config.get("batch_size"),
                torch_dtype=self.embedding_config.get("torch_dtype"),
            )

        elif self.embedding_model not in ["default", "openai", "gemini"]:
            # Treat any other value as a HuggingFace model name
            return HuggingFaceEmbeddingFunction(
                model_name=self.embedding_model,
                device=self.embedding_config.get("device"),
                batch_size=self.embedding_config.get("batch_size"),
                torch_dtype=self.embedding_config.get("torch_dtype"),
            )

        else:
            # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
            ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
            ef.max_input_tokens = 256  # all-MiniLM-L6-v2 max_seq_length
            return ef

    @property
    def embedding_max_tokens(self) -> int:
        """Maximum input tokens supported by the configured embedding model."""
        return getattr(self.embedding_function, "max_input_tokens", 8000)

    def truncate_text(self, text: str, max_tokens: int | None = None) -> str:
        """Truncate text using the embedding function's model-aware tokenizer.

        Falls back to tiktoken cl100k_base or character estimation if the
        embedding function does not provide a truncate method.
        """
        if max_tokens is None:
            max_tokens = self.embedding_max_tokens
        if hasattr(self.embedding_function, 'truncate'):
            return self.embedding_function.truncate(text, max_tokens)
        # Fallback for default ChromaDB embedding function
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text, disallowed_special=())
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = enc.decode(tokens)
        except Exception:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                text = text[:max_chars]
        return text

    def add_documents(self,
                     documents: list[str],
                     metadatas: list[dict[str, Any]],
                     ids: list[str]) -> None:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def upsert_documents(self,
                        documents: list[str],
                        metadatas: list[dict[str, Any]],
                        ids: list[str]) -> None:
        """
        Upsert (update or insert) documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Upserted {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting documents to ChromaDB: {e}")
            raise

    def search(self,
               query_texts: list[str],
               n_results: int = 10,
               where: dict[str, Any] | None = None,
               where_document: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            Search results from ChromaDB
        """
        try:
            query_kwargs = {
                "n_results": n_results,
                "where": where,
                "where_document": where_document,
            }

            # Use embed_query for our custom embedding functions that implement
            # correct query-time task types (e.g. Gemini retrieval_query).
            # Do NOT use embed_query on ChromaDB's DefaultEmbeddingFunction —
            # its embed_query returns chunked results, not a single vector.
            _is_custom_ef = isinstance(
                self.embedding_function,
                (OpenAIEmbeddingFunction, GeminiEmbeddingFunction, HuggingFaceEmbeddingFunction),
            )
            if _is_custom_ef and hasattr(self.embedding_function, 'embed_query') and query_texts:
                query_embeddings = []
                for qt in query_texts:
                    emb = self.embedding_function.embed_query(qt)
                    # Ensure plain Python floats (some providers return numpy)
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    query_embeddings.append(emb)
                query_kwargs["query_embeddings"] = query_embeddings
            else:
                query_kwargs["query_texts"] = query_texts

            results = self.collection.query(**query_kwargs)
            logger.info(f"Semantic search returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            raise

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents from the collection.

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise

    def delete_documents_for_item_keys(self, item_keys: list[str]) -> int:
        """Delete all Chroma documents that belong to the given Zotero items."""
        ids_to_delete = self.get_document_ids_for_item_keys(item_keys)
        if ids_to_delete:
            self.delete_documents(sorted(ids_to_delete))
        return len(ids_to_delete)

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }

    def reset_collection(self) -> None:
        """Reset (clear) the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get metadata for a document if it exists.

        Args:
            doc_id: Document ID to look up

        Returns:
            Metadata dictionary if document exists, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["metadatas"])
            if result['ids'] and result['metadatas']:
                return result['metadatas'][0]
            return None
        except Exception:
            return None

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ids that already exist in the collection."""
        if not ids:
            return set()
        try:
            result = self.collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()

    def get_all_ids(self) -> set[str]:
        """Return every id currently stored in the collection.

        Used by incremental sync to compute deletions: items in the local
        collection but no longer present in the Zotero library.
        """
        try:
            result = self.collection.get(include=[])
            return set(result.get("ids", []))
        except Exception as e:
            logger.error(f"Error listing collection ids: {e}")
            return set()

    def get_document_ids_for_item_keys(self, item_keys: list[str]) -> set[str]:
        """Return all Chroma document ids belonging to the given Zotero items."""
        ids: set[str] = set()
        for item_key in item_keys:
            if not item_key:
                continue
            try:
                result = self.collection.get(where={"item_key": item_key}, include=[])
                ids.update(result.get("ids", []))
            except Exception:
                pass
            # Legacy one-document-per-item collections used the Zotero key as
            # the Chroma id. Include it so chunked reindexing can replace old
            # entries cleanly.
            if self.document_exists(item_key):
                ids.add(item_key)
        return ids

    def get_all_item_keys(self) -> set[str]:
        """Return Zotero item keys represented in the collection."""
        try:
            result = self.collection.get(include=["metadatas"])
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", []) or []
            item_keys: set[str] = set()
            for idx, doc_id in enumerate(ids):
                metadata = metadatas[idx] if idx < len(metadatas) else None
                item_key = metadata.get("item_key") if isinstance(metadata, dict) else None
                item_keys.add(item_key or doc_id)
            return item_keys
        except Exception as e:
            logger.error(f"Error listing collection item keys: {e}")
            return set()


def create_chroma_client(config_path: str | None = None) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured ChromaClient instance
    """
    # Default configuration
    config = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {}
    }

    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config.get("semantic_search", {}))
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Load configuration from environment variables
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model:
        config["embedding_model"] = env_embedding_model

    # Merge embedding config from environment (config.json wins, env fills gaps).
    # Precedence: explicit config.json value > env var > hardcoded default.
    # Previous code unconditionally REPLACED config["embedding_config"] with env
    # values, silently dropping model_name from config.json whenever any
    # provider env var (e.g. GOOGLE_API_KEY leaked from another tool) was set.
    if config["embedding_model"] == "openai":
        ec = dict(config.get("embedding_config") or {})
        if not ec.get("api_key"):
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                ec["api_key"] = env_key
        if not ec.get("model_name"):
            ec["model_name"] = os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        if not ec.get("base_url"):
            env_base = os.getenv("OPENAI_BASE_URL")
            if env_base:
                ec["base_url"] = env_base
        if ec.get("api_key"):
            config["embedding_config"] = ec

    elif config["embedding_model"] == "gemini":
        ec = dict(config.get("embedding_config") or {})
        if not ec.get("api_key"):
            env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if env_key:
                ec["api_key"] = env_key
        if not ec.get("model_name"):
            ec["model_name"] = os.getenv(
                "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
            )
        if not ec.get("base_url"):
            env_base = os.getenv("GEMINI_BASE_URL")
            if env_base:
                ec["base_url"] = env_base
        if ec.get("api_key"):
            config["embedding_config"] = ec

    return ChromaClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"]
    )
