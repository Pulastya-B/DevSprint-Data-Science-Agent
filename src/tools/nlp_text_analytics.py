"""
NLP & Text Analytics Tools

Advanced natural language processing tools for text analysis, topic modeling,
named entity recognition, sentiment analysis, and text similarity.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

# Core NLP
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    pass

# Advanced NLP (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Basic NLP
try:
    from textblob import TextBlob
except ImportError:
    pass

import re
from collections import Counter


def perform_topic_modeling(
    data: pl.DataFrame,
    text_column: str,
    n_topics: int = 5,
    method: str = "lda",
    n_top_words: int = 10,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform topic modeling on text data using LDA, NMF, or BERTopic.
    
    Args:
        data: Input DataFrame
        text_column: Column containing text data
        n_topics: Number of topics to extract
        method: Topic modeling method ('lda', 'nmf', 'bertopic')
        n_top_words: Number of top words per topic
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        ngram_range: Range of n-grams to extract
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for the chosen method
    
    Returns:
        Dictionary containing topics, document-topic distributions, and metrics
    """
    print(f"🔍 Performing topic modeling using {method.upper()}...")
    
    # Validate input
    if text_column not in data.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    
    # Extract text and clean
    texts = data[text_column].to_list()
    texts = [str(t) if t is not None else "" for t in texts]
    
    # Filter out empty texts
    valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 0]
    texts_clean = [texts[i] for i in valid_indices]
    
    if len(texts_clean) < n_topics:
        raise ValueError(f"Not enough documents ({len(texts_clean)}) for {n_topics} topics")
    
    result = {
        "method": method,
        "n_topics": n_topics,
        "n_documents": len(texts_clean),
        "topics": [],
        "document_topics": None,
        "topic_coherence": None
    }
    
    try:
        if method == "bertopic" and BERTOPIC_AVAILABLE:
            # BERTopic - transformer-based topic modeling
            print("  Using BERTopic (transformer-based)...")
            
            model = BERTopic(
                nr_topics=n_topics,
                language="english",
                calculate_probabilities=True,
                verbose=False,
                **kwargs
            )
            
            topics_assigned, probabilities = model.fit_transform(texts_clean)
            
            # Extract topic information
            topic_info = model.get_topic_info()
            
            for topic_id in range(n_topics):
                if topic_id in model.get_topics():
                    topic_words = model.get_topic(topic_id)[:n_top_words]
                    result["topics"].append({
                        "topic_id": topic_id,
                        "words": [word for word, score in topic_words],
                        "scores": [float(score) for word, score in topic_words],
                        "size": int(topic_info[topic_info['Topic'] == topic_id]['Count'].iloc[0])
                    })
            
            # Document-topic distributions
            result["document_topics"] = probabilities.tolist() if probabilities is not None else None
            result["topic_assignments"] = topics_assigned.tolist()
            
        elif method in ["lda", "nmf"]:
            # Traditional topic modeling with sklearn
            print(f"  Using {method.upper()} with TF-IDF/Count vectorization...")
            
            # Vectorization
            if method == "lda":
                vectorizer = CountVectorizer(
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=ngram_range,
                    stop_words='english',
                    max_features=kwargs.get('max_features', 1000)
                )
            else:  # nmf
                vectorizer = TfidfVectorizer(
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=ngram_range,
                    stop_words='english',
                    max_features=kwargs.get('max_features', 1000)
                )
            
            doc_term_matrix = vectorizer.fit_transform(texts_clean)
            feature_names = vectorizer.get_feature_names_out()
            
            # Topic modeling
            if method == "lda":
                model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=random_state,
                    max_iter=kwargs.get('max_iter', 20),
                    learning_method='online',
                    n_jobs=-1
                )
            else:  # nmf
                model = NMF(
                    n_components=n_topics,
                    random_state=random_state,
                    max_iter=kwargs.get('max_iter', 200),
                    init='nndsvda'
                )
            
            doc_topic_dist = model.fit_transform(doc_term_matrix)
            
            # Extract topics
            for topic_idx, topic in enumerate(model.components_):
                top_indices = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                top_scores = [float(topic[i]) for i in top_indices]
                
                result["topics"].append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "scores": top_scores,
                    "size": int((doc_topic_dist.argmax(axis=1) == topic_idx).sum())
                })
            
            # Document-topic distributions
            result["document_topics"] = doc_topic_dist.tolist()
            
            # Topic assignments (most probable topic per document)
            result["topic_assignments"] = doc_topic_dist.argmax(axis=1).tolist()
            
            # Calculate perplexity for LDA
            if method == "lda":
                result["perplexity"] = float(model.perplexity(doc_term_matrix))
                result["log_likelihood"] = float(model.score(doc_term_matrix))
            
            # Vocabulary size
            result["vocabulary_size"] = len(feature_names)
            
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'lda', 'nmf', or 'bertopic'")
        
        # Calculate topic diversity (unique words across topics)
        all_topic_words = set()
        total_topic_words = 0
        for topic in result["topics"]:
            all_topic_words.update(topic["words"])
            total_topic_words += len(topic["words"])
        
        result["topic_diversity"] = len(all_topic_words) / total_topic_words if total_topic_words > 0 else 0
        
        # Summary statistics
        result["summary"] = {
            "total_topics": len(result["topics"]),
            "avg_topic_size": np.mean([t["size"] for t in result["topics"]]),
            "topic_diversity": result["topic_diversity"]
        }
        
        print(f"✅ Topic modeling complete! Found {len(result['topics'])} topics")
        print(f"   Topic diversity: {result['topic_diversity']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during topic modeling: {str(e)}")
        raise


def perform_named_entity_recognition(
    data: pl.DataFrame,
    text_column: str,
    model: str = "en_core_web_sm",
    entity_types: Optional[List[str]] = None,
    min_confidence: float = 0.0
) -> Dict[str, Any]:
    """
    Perform named entity recognition to extract people, organizations, locations, etc.
    
    Args:
        data: Input DataFrame
        text_column: Column containing text data
        model: spaCy model to use ('en_core_web_sm', 'en_core_web_md', 'en_core_web_lg')
        entity_types: List of entity types to extract (e.g., ['PERSON', 'ORG', 'GPE'])
                     If None, extracts all types
        min_confidence: Minimum confidence score for entity extraction (0.0-1.0)
    
    Returns:
        Dictionary containing extracted entities, counts, and statistics
    """
    print(f"🔍 Performing named entity recognition with spaCy...")
    
    if not SPACY_AVAILABLE:
        # Fallback to basic pattern matching
        print("⚠️  spaCy not available. Using basic pattern matching...")
        return _perform_ner_basic(data, text_column)
    
    # Validate input
    if text_column not in data.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    
    try:
        # Load spaCy model
        try:
            nlp = spacy.load(model)
        except OSError:
            print(f"⚠️  Model '{model}' not found. Attempting to download...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            nlp = spacy.load(model)
        
        # Extract text
        texts = data[text_column].to_list()
        texts = [str(t) if t is not None else "" for t in texts]
        
        # Process documents
        all_entities = []
        entity_counts = Counter()
        entity_by_type = {}
        
        print(f"  Processing {len(texts)} documents...")
        
        for doc_idx, text in enumerate(texts):
            if len(text.strip()) == 0:
                continue
            
            doc = nlp(text)
            
            for ent in doc.ents:
                # Filter by entity type if specified
                if entity_types and ent.label_ not in entity_types:
                    continue
                
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "document_id": doc_idx
                }
                
                all_entities.append(entity_info)
                entity_counts[(ent.text, ent.label_)] += 1
                
                if ent.label_ not in entity_by_type:
                    entity_by_type[ent.label_] = []
                entity_by_type[ent.label_].append(ent.text)
        
        # Aggregate results
        result = {
            "total_entities": len(all_entities),
            "unique_entities": len(entity_counts),
            "entities": all_entities,
            "entity_counts": [
                {"text": text, "label": label, "count": count}
                for (text, label), count in entity_counts.most_common(100)
            ],
            "by_type": {}
        }
        
        # Statistics by entity type
        for entity_type, entities in entity_by_type.items():
            type_counter = Counter(entities)
            result["by_type"][entity_type] = {
                "total": len(entities),
                "unique": len(type_counter),
                "top_entities": [
                    {"text": text, "count": count}
                    for text, count in type_counter.most_common(10)
                ]
            }
        
        print(f"✅ NER complete! Found {result['total_entities']} entities")
        print(f"   Unique entities: {result['unique_entities']}")
        print(f"   Entity types: {', '.join(result['by_type'].keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during NER: {str(e)}")
        raise


def _perform_ner_basic(data: pl.DataFrame, text_column: str) -> Dict[str, Any]:
    """Fallback NER using basic pattern matching when spaCy is not available."""
    
    texts = data[text_column].to_list()
    texts = [str(t) if t is not None else "" for t in texts]
    
    # Basic patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    emails = []
    urls = []
    phones = []
    
    for text in texts:
        emails.extend(re.findall(email_pattern, text))
        urls.extend(re.findall(url_pattern, text))
        phones.extend(re.findall(phone_pattern, text))
    
    return {
        "method": "basic_pattern_matching",
        "total_entities": len(emails) + len(urls) + len(phones),
        "by_type": {
            "EMAIL": {"total": len(emails), "unique": len(set(emails)), "examples": list(set(emails))[:10]},
            "URL": {"total": len(urls), "unique": len(set(urls)), "examples": list(set(urls))[:10]},
            "PHONE": {"total": len(phones), "unique": len(set(phones)), "examples": list(set(phones))[:10]}
        },
        "note": "Install spaCy for advanced NER: pip install spacy && python -m spacy download en_core_web_sm"
    }


def analyze_sentiment_advanced(
    data: pl.DataFrame,
    text_column: str,
    method: str = "transformer",
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    aspects: Optional[List[str]] = None,
    detect_emotions: bool = True
) -> Dict[str, Any]:
    """
    Perform advanced sentiment analysis with aspect-based sentiment and emotion detection.
    
    Args:
        data: Input DataFrame
        text_column: Column containing text data
        method: Analysis method ('transformer', 'textblob', 'vader')
        model_name: Transformer model for sentiment analysis
        aspects: List of aspects for aspect-based sentiment (e.g., ['price', 'quality'])
        detect_emotions: Whether to detect emotions (joy, anger, sadness, etc.)
    
    Returns:
        Dictionary containing sentiment scores, emotions, and statistics
    """
    print(f"🔍 Performing advanced sentiment analysis...")
    
    # Validate input
    if text_column not in data.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    
    # Extract text
    texts = data[text_column].to_list()
    texts = [str(t) if t is not None else "" for t in texts]
    texts_clean = [t for t in texts if len(t.strip()) > 0]
    
    result = {
        "method": method,
        "n_documents": len(texts_clean),
        "sentiments": [],
        "statistics": {}
    }
    
    try:
        if method == "transformer" and TRANSFORMERS_AVAILABLE:
            print(f"  Using transformer model: {model_name}")
            
            # Sentiment analysis pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                truncation=True,
                max_length=512
            )
            
            # Process in batches
            batch_size = 32
            all_sentiments = []
            
            for i in range(0, len(texts_clean), batch_size):
                batch = texts_clean[i:i+batch_size]
                batch_results = sentiment_pipeline(batch)
                all_sentiments.extend(batch_results)
            
            result["sentiments"] = [
                {
                    "label": s["label"],
                    "score": float(s["score"]),
                    "text": texts_clean[i][:100]  # First 100 chars
                }
                for i, s in enumerate(all_sentiments)
            ]
            
            # Emotion detection
            if detect_emotions:
                try:
                    emotion_pipeline = pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        truncation=True,
                        max_length=512
                    )
                    
                    emotions = []
                    for i in range(0, len(texts_clean), batch_size):
                        batch = texts_clean[i:i+batch_size]
                        batch_emotions = emotion_pipeline(batch)
                        emotions.extend(batch_emotions)
                    
                    result["emotions"] = [
                        {"emotion": e["label"], "score": float(e["score"])}
                        for e in emotions
                    ]
                    
                    # Emotion distribution
                    emotion_counts = Counter([e["label"] for e in emotions])
                    result["emotion_distribution"] = dict(emotion_counts)
                    
                except Exception as e:
                    print(f"⚠️  Emotion detection failed: {str(e)}")
                    result["emotions"] = None
            
        else:
            # Fallback to TextBlob
            print("  Using TextBlob for sentiment analysis...")
            
            sentiments = []
            for text in texts_clean:
                blob = TextBlob(text)
                sentiments.append({
                    "polarity": blob.sentiment.polarity,
                    "subjectivity": blob.sentiment.subjectivity,
                    "label": "POSITIVE" if blob.sentiment.polarity > 0 else "NEGATIVE" if blob.sentiment.polarity < 0 else "NEUTRAL",
                    "text": text[:100]
                })
            
            result["sentiments"] = sentiments
        
        # Aspect-based sentiment
        if aspects:
            print(f"  Analyzing aspect-based sentiment for: {', '.join(aspects)}")
            result["aspect_sentiments"] = _extract_aspect_sentiments(texts_clean, aspects)
        
        # Calculate statistics
        if method == "transformer":
            sentiment_counts = Counter([s["label"] for s in result["sentiments"]])
            result["statistics"] = {
                "sentiment_distribution": dict(sentiment_counts),
                "positive_ratio": sentiment_counts.get("POSITIVE", 0) / len(texts_clean),
                "negative_ratio": sentiment_counts.get("NEGATIVE", 0) / len(texts_clean),
                "avg_confidence": np.mean([s["score"] for s in result["sentiments"]])
            }
        else:
            polarities = [s["polarity"] for s in result["sentiments"]]
            result["statistics"] = {
                "avg_polarity": np.mean(polarities),
                "std_polarity": np.std(polarities),
                "positive_ratio": sum(1 for p in polarities if p > 0) / len(polarities),
                "negative_ratio": sum(1 for p in polarities if p < 0) / len(polarities),
                "neutral_ratio": sum(1 for p in polarities if p == 0) / len(polarities)
            }
        
        print(f"✅ Sentiment analysis complete!")
        print(f"   Distribution: {result['statistics'].get('sentiment_distribution', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {str(e)}")
        raise


def _extract_aspect_sentiments(texts: List[str], aspects: List[str]) -> Dict[str, Any]:
    """Extract sentiment for specific aspects in text."""
    
    aspect_sentiments = {aspect: [] for aspect in aspects}
    
    for text in texts:
        text_lower = text.lower()
        
        for aspect in aspects:
            # Find sentences containing the aspect
            sentences = text.split('.')
            aspect_sentences = [s for s in sentences if aspect.lower() in s.lower()]
            
            if aspect_sentences:
                # Analyze sentiment of aspect sentences
                for sentence in aspect_sentences:
                    blob = TextBlob(sentence)
                    aspect_sentiments[aspect].append({
                        "text": sentence.strip(),
                        "polarity": blob.sentiment.polarity,
                        "subjectivity": blob.sentiment.subjectivity
                    })
    
    # Aggregate aspect sentiments
    result = {}
    for aspect, sentiments in aspect_sentiments.items():
        if sentiments:
            polarities = [s["polarity"] for s in sentiments]
            result[aspect] = {
                "count": len(sentiments),
                "avg_polarity": np.mean(polarities),
                "positive_mentions": sum(1 for p in polarities if p > 0),
                "negative_mentions": sum(1 for p in polarities if p < 0),
                "examples": sentiments[:5]
            }
        else:
            result[aspect] = {"count": 0, "avg_polarity": 0.0}
    
    return result


def perform_text_similarity(
    data: pl.DataFrame,
    text_column: str,
    query_text: Optional[str] = None,
    method: str = "cosine",
    top_k: int = 10,
    use_embeddings: bool = False,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Calculate text similarity using cosine, Jaccard, or semantic embeddings.
    
    Args:
        data: Input DataFrame
        text_column: Column containing text data
        query_text: Query text to find similar documents (if None, computes pairwise)
        method: Similarity method ('cosine', 'jaccard', 'semantic')
        top_k: Number of top similar documents to return
        use_embeddings: Whether to use transformer embeddings (for semantic similarity)
        model_name: Model for semantic embeddings
    
    Returns:
        Dictionary containing similarity scores and top matches
    """
    print(f"🔍 Calculating text similarity using {method} method...")
    
    # Validate input
    if text_column not in data.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    
    # Extract text
    texts = data[text_column].to_list()
    texts = [str(t) if t is not None else "" for t in texts]
    
    result = {
        "method": method,
        "n_documents": len(texts),
        "query_text": query_text,
        "similarities": []
    }
    
    try:
        if method == "semantic" and use_embeddings and TRANSFORMERS_AVAILABLE:
            print(f"  Using semantic embeddings: {model_name}")
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            def get_embedding(text: str) -> np.ndarray:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Mean pooling
                return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Get embeddings
            if query_text:
                query_embedding = get_embedding(query_text)
                text_embeddings = np.array([get_embedding(t) for t in texts])
                
                # Calculate cosine similarity
                similarities = cosine_similarity([query_embedding], text_embeddings)[0]
                
                # Top K
                top_indices = similarities.argsort()[-top_k:][::-1]
                result["similarities"] = [
                    {
                        "document_id": int(idx),
                        "text": texts[idx][:200],
                        "score": float(similarities[idx])
                    }
                    for idx in top_indices
                ]
            else:
                # Pairwise similarity
                text_embeddings = np.array([get_embedding(t) for t in texts])
                similarity_matrix = cosine_similarity(text_embeddings)
                result["similarity_matrix"] = similarity_matrix.tolist()
                result["avg_similarity"] = float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
        
        elif method == "cosine":
            print("  Using TF-IDF with cosine similarity...")
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            if query_text:
                all_texts = [query_text] + texts
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                
                # Similarity between query and all documents
                similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                
                # Top K
                top_indices = similarities.argsort()[-top_k:][::-1]
                result["similarities"] = [
                    {
                        "document_id": int(idx),
                        "text": texts[idx][:200],
                        "score": float(similarities[idx])
                    }
                    for idx in top_indices
                ]
            else:
                # Pairwise similarity
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                result["similarity_matrix"] = similarity_matrix.tolist()
                result["avg_similarity"] = float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
        
        elif method == "jaccard":
            print("  Using Jaccard similarity...")
            
            def jaccard_similarity(text1: str, text2: str) -> float:
                set1 = set(text1.lower().split())
                set2 = set(text2.lower().split())
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                return intersection / union if union > 0 else 0.0
            
            if query_text:
                similarities = [jaccard_similarity(query_text, text) for text in texts]
                
                # Top K
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                result["similarities"] = [
                    {
                        "document_id": int(idx),
                        "text": texts[idx][:200],
                        "score": float(similarities[idx])
                    }
                    for idx in top_indices
                ]
            else:
                # Pairwise similarity
                n = len(texts)
                similarity_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        sim = jaccard_similarity(texts[i], texts[j])
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                
                result["similarity_matrix"] = similarity_matrix.tolist()
                result["avg_similarity"] = float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
        
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cosine', 'jaccard', or 'semantic'")
        
        print(f"✅ Similarity calculation complete!")
        if result.get("similarities"):
            print(f"   Top similarity score: {result['similarities'][0]['score']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during similarity calculation: {str(e)}")
        raise
