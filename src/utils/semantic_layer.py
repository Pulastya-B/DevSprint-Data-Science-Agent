"""
Semantic Layer using SBERT for Column Understanding and Agent Routing

Provides semantic understanding of dataset columns and agent intent matching
using sentence-transformers embeddings.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import polars as pl
from pathlib import Path
import json

# SBERT for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Install with: pip install sentence-transformers")

# Sklearn for similarity
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SemanticLayer:
    """
    Semantic understanding layer using SBERT embeddings.
    
    Features:
    - Column semantic embedding (name + sample values + dtype)
    - Semantic column matching (find similar columns)
    - Agent intent routing (semantic task → agent mapping)
    - Target column inference (semantic similarity to "target")
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic layer with SBERT model.
        
        Args:
            model_name: Sentence-transformer model name
                - all-MiniLM-L6-v2: Fast, 384 dims (recommended)
                - all-mpnet-base-v2: Better quality, 768 dims, slower
                - paraphrase-MiniLM-L6-v2: Good for short texts
        """
        self.model_name = model_name
        self.model = None
        self.enabled = SBERT_AVAILABLE and SKLEARN_AVAILABLE
        
        if self.enabled:
            try:
                print(f"🧠 Loading SBERT model: {model_name}...")
                # Try loading with trust_remote_code for better compatibility
                self.model = SentenceTransformer(model_name, trust_remote_code=True)
                # Use GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                    print("✅ SBERT loaded on GPU")
                else:
                    print("✅ SBERT loaded on CPU")
            except Exception as e:
                print(f"⚠️ Failed to load SBERT model: {e}")
                print(f"   Falling back to keyword-based routing (semantic features disabled)")
                self.enabled = False
        else:
            print("⚠️ SBERT semantic layer disabled (missing dependencies)")
    
    def encode_column(self, column_name: str, dtype: str, 
                      sample_values: Optional[List[Any]] = None,
                      stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Create semantic embedding for a column.
        
        Combines column name, data type, sample values, and stats into
        a text description that captures the column's semantic meaning.
        
        Args:
            column_name: Name of the column
            dtype: Data type (Int64, Float64, Utf8, etc.)
            sample_values: Sample values from the column
            stats: Optional statistics (mean, min, max, etc.)
        
        Returns:
            Embedding vector (numpy array)
        
        Example:
            >>> encode_column("annual_salary", "Float64", [50000, 75000], {"mean": 65000})
            >>> # Returns embedding for "annual_salary (Float64 numeric): values like 50000, 75000, mean 65000"
        """
        if not self.enabled:
            return np.zeros(384)  # Dummy embedding
        
        # Build semantic description
        description_parts = [f"Column name: {column_name}"]
        
        # Add type information
        type_desc = self._interpret_dtype(dtype)
        description_parts.append(f"Type: {type_desc}")
        
        # Add sample values
        if sample_values:
            # Format samples nicely
            samples_str = ", ".join([str(v)[:50] for v in sample_values[:5] if v is not None])
            description_parts.append(f"Example values: {samples_str}")
        
        # Add statistics
        if stats:
            if 'mean' in stats and stats['mean'] is not None:
                description_parts.append(f"Mean: {stats['mean']:.2f}")
            if 'unique_count' in stats and stats['unique_count'] is not None:
                description_parts.append(f"Unique values: {stats['unique_count']}")
            if 'null_percentage' in stats and stats['null_percentage'] is not None:
                description_parts.append(f"Missing: {stats['null_percentage']:.1f}%")
        
        # Combine into single text
        text = ". ".join(description_parts)
        
        # Generate embedding
        try:
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding
        except Exception as e:
            print(f"⚠️ Error encoding column {column_name}: {e}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def _interpret_dtype(self, dtype: str) -> str:
        """Convert polars dtype to human-readable description."""
        dtype_lower = str(dtype).lower()
        
        if 'int' in dtype_lower or 'float' in dtype_lower:
            return "numeric continuous or count data"
        elif 'bool' in dtype_lower:
            return "boolean flag"
        elif 'utf8' in dtype_lower or 'str' in dtype_lower:
            return "text or categorical label"
        elif 'date' in dtype_lower or 'time' in dtype_lower:
            return "temporal timestamp"
        else:
            return "data values"
    
    def find_similar_columns(self, query_column: str, column_embeddings: Dict[str, np.ndarray],
                            top_k: int = 3, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find columns semantically similar to query column.
        
        Use case: Detect duplicates or related columns
        Example: "Salary" → finds ["Annual_Income", "Compensation", "Pay"]
        
        Args:
            query_column: Column name to search for
            column_embeddings: Dict mapping column names to their embeddings
            top_k: Number of similar columns to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of (column_name, similarity_score) tuples
        """
        if not self.enabled or query_column not in column_embeddings:
            return []
        
        query_emb = column_embeddings[query_column].reshape(1, -1)
        
        similarities = []
        for col_name, col_emb in column_embeddings.items():
            if col_name == query_column:
                continue
            
            sim = cosine_similarity(query_emb, col_emb.reshape(1, -1))[0][0]
            if sim >= threshold:
                similarities.append((col_name, float(sim)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def infer_target_column(self, column_embeddings: Dict[str, np.ndarray],
                           task_description: str) -> Optional[Tuple[str, float]]:
        """
        Infer which column is likely the target/label for prediction.
        
        Uses semantic similarity between column descriptions and task description.
        
        Args:
            column_embeddings: Dict mapping column names to embeddings
            task_description: User's task description
        
        Returns:
            (column_name, confidence_score) or None
        
        Example:
            >>> infer_target_column(embeddings, "predict house prices")
            >>> ("Price", 0.85)  # High confidence "Price" is target
        """
        if not self.enabled:
            return None
        
        # Encode task description
        task_emb = self.model.encode(task_description, convert_to_numpy=True, show_progress_bar=False)
        task_emb = task_emb.reshape(1, -1)
        
        # Find column with highest similarity to task
        best_col = None
        best_score = 0.0
        
        for col_name, col_emb in column_embeddings.items():
            sim = cosine_similarity(task_emb, col_emb.reshape(1, -1))[0][0]
            if sim > best_score:
                best_score = sim
                best_col = col_name
        
        # Only return if confidence is reasonable
        if best_score >= 0.4:  # Threshold for target inference
            return (best_col, float(best_score))
        
        return None
    
    def route_to_agent(self, task_description: str, 
                       agent_descriptions: Dict[str, str]) -> Tuple[str, float]:
        """
        Route task to appropriate specialist agent using semantic similarity.
        
        Replaces keyword-based routing with semantic understanding.
        
        Args:
            task_description: User's task description
            agent_descriptions: Dict mapping agent_key → agent description
        
        Returns:
            (agent_key, confidence_score)
        
        Example:
            >>> route_to_agent("build a predictive model", {
            ...     "modeling_agent": "Expert in ML training and models",
            ...     "viz_agent": "Expert in visualizations"
            ... })
            >>> ("modeling_agent", 0.92)
        """
        if not self.enabled:
            # Fallback to first agent
            return list(agent_descriptions.keys())[0], 0.5
        
        # Encode task
        task_emb = self.model.encode(task_description, convert_to_numpy=True, show_progress_bar=False)
        task_emb = task_emb.reshape(1, -1)
        
        # Encode agent descriptions
        best_agent = None
        best_score = 0.0
        
        for agent_key, agent_desc in agent_descriptions.items():
            agent_emb = self.model.encode(agent_desc, convert_to_numpy=True, show_progress_bar=False)
            agent_emb = agent_emb.reshape(1, -1)
            
            sim = cosine_similarity(task_emb, agent_emb)[0][0]
            if sim > best_score:
                best_score = sim
                best_agent = agent_key
        
        return best_agent, float(best_score)
    
    def semantic_column_match(self, target_name: str, available_columns: List[str],
                             threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Find best matching column for a target name using fuzzy semantic matching.
        
        Better than string fuzzy matching because it understands synonyms:
        - "salary" matches "annual_income", "compensation", "pay"
        - "target" matches "label", "class", "outcome"
        
        Args:
            target_name: Column name to find (might not exist exactly)
            available_columns: List of actual column names in dataset
            threshold: Minimum similarity to consider a match
        
        Returns:
            (matched_column, confidence) or None
        
        Example:
            >>> semantic_column_match("salary", ["Annual_Income", "Name", "Age"])
            >>> ("Annual_Income", 0.78)
        """
        if not self.enabled:
            # Fallback to exact match
            if target_name in available_columns:
                return (target_name, 1.0)
            return None
        
        # Encode target
        target_emb = self.model.encode(target_name, convert_to_numpy=True, show_progress_bar=False)
        target_emb = target_emb.reshape(1, -1)
        
        # Find best match
        best_col = None
        best_score = 0.0
        
        for col in available_columns:
            col_emb = self.model.encode(col, convert_to_numpy=True, show_progress_bar=False)
            col_emb = col_emb.reshape(1, -1)
            
            sim = cosine_similarity(target_emb, col_emb)[0][0]
            if sim > best_score:
                best_score = sim
                best_col = col
        
        if best_score >= threshold:
            return (best_col, float(best_score))
        
        return None
    
    def enrich_dataset_info(self, dataset_info: Dict[str, Any], 
                           file_path: str, sample_size: int = 100) -> Dict[str, Any]:
        """
        Enrich dataset_info with semantic column embeddings.
        
        Adds 'column_embeddings' and 'semantic_insights' to dataset_info.
        
        Args:
            dataset_info: Dataset info from schema_extraction
            file_path: Path to CSV file
            sample_size: Number of rows to sample for encoding
        
        Returns:
            Enhanced dataset_info with semantic layer
        """
        if not self.enabled:
            return dataset_info
        
        try:
            # Load dataset
            df = pl.read_csv(file_path, n_rows=sample_size)
            
            column_embeddings = {}
            
            for col_name, col_info in dataset_info['columns'].items():
                # Get sample values
                sample_values = df[col_name].head(5).to_list()
                
                # Create embedding
                embedding = self.encode_column(
                    column_name=col_name,
                    dtype=col_info['dtype'],
                    sample_values=sample_values,
                    stats={
                        'unique_count': col_info.get('unique_count'),
                        'missing_pct': col_info.get('missing_pct'),
                        'mean': col_info.get('mean')
                    }
                )
                
                column_embeddings[col_name] = embedding
            
            # Add to dataset_info
            dataset_info['column_embeddings'] = column_embeddings
            
            # Detect similar columns (potential duplicates)
            similar_pairs = []
            cols = list(column_embeddings.keys())
            for i, col1 in enumerate(cols):
                similar = self.find_similar_columns(col1, column_embeddings, top_k=1, threshold=0.75)
                if similar:
                    similar_pairs.append((col1, similar[0][0], similar[0][1]))
            
            dataset_info['semantic_insights'] = {
                'similar_columns': similar_pairs,
                'total_columns_embedded': len(column_embeddings)
            }
            
            print(f"🧠 Semantic layer: Embedded {len(column_embeddings)} columns")
            if similar_pairs:
                print(f"   Found {len(similar_pairs)} similar column pairs (potential duplicates)")
            
        except Exception as e:
            print(f"⚠️ Error enriching dataset with semantic layer: {e}")
        
        return dataset_info


# Global semantic layer instance (lazy loaded)
_semantic_layer = None

def get_semantic_layer() -> SemanticLayer:
    """Get or create global semantic layer instance."""
    global _semantic_layer
    if _semantic_layer is None:
        _semantic_layer = SemanticLayer()
    return _semantic_layer
