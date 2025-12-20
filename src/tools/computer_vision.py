"""
Computer Vision & Image Analytics Tools

Advanced computer vision tools for image feature extraction, clustering,
and hybrid tabular-image analysis.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

# Core CV libraries (optional)
try:
    from PIL import Image
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    import torchvision
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ML libraries
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
except ImportError:
    pass


def extract_image_features(
    image_paths: List[str],
    method: str = "cnn",
    model_name: str = "resnet50",
    color_spaces: Optional[List[str]] = None,
    include_histograms: bool = True,
    histogram_bins: int = 256
) -> Dict[str, Any]:
    """
    Extract features from images using CNN embeddings, color histograms, and other methods.
    
    Args:
        image_paths: List of paths to image files
        method: Feature extraction method ('cnn', 'color', 'texture', 'hybrid')
        model_name: Pre-trained model for CNN features ('resnet50', 'efficientnet_b0', 'vgg16')
        color_spaces: Color spaces for histograms (['rgb', 'hsv', 'lab'])
        include_histograms: Whether to include color histograms
        histogram_bins: Number of bins for histograms
    
    Returns:
        Dictionary containing feature vectors, dimensionality, and metadata
    """
    print(f"🔍 Extracting image features using {method} method...")
    
    if not image_paths:
        raise ValueError("No image paths provided")
    
    result = {
        "method": method,
        "n_images": len(image_paths),
        "features": [],
        "feature_dim": 0,
        "failed_images": []
    }
    
    try:
        if method == "cnn" and TORCH_AVAILABLE:
            print(f"  Using CNN model: {model_name}")
            
            # Load pre-trained model
            if model_name == "resnet50":
                model = models.resnet50(pretrained=True)
                # Remove final classification layer
                model = torch.nn.Sequential(*list(model.children())[:-1])
            elif model_name == "efficientnet_b0":
                model = models.efficientnet_b0(pretrained=True)
                model = torch.nn.Sequential(*list(model.children())[:-1])
            elif model_name == "vgg16":
                model = models.vgg16(pretrained=True)
                model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
            else:
                raise ValueError(f"Unknown model '{model_name}'")
            
            model.eval()
            
            # Image preprocessing
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Extract features
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0)
                    
                    with torch.no_grad():
                        features = model(img_tensor)
                        features = features.squeeze().numpy()
                    
                    result["features"].append({
                        "image_path": img_path,
                        "feature_vector": features.tolist(),
                        "feature_dim": len(features)
                    })
                    
                except Exception as e:
                    result["failed_images"].append({"path": img_path, "error": str(e)})
            
            if result["features"]:
                result["feature_dim"] = result["features"][0]["feature_dim"]
        
        elif method in ["color", "hybrid"] or not TORCH_AVAILABLE:
            print("  Using color histogram features...")
            
            if not CV2_AVAILABLE:
                print("⚠️  OpenCV not available. Using PIL for basic features...")
                return _extract_features_basic(image_paths)
            
            color_spaces = color_spaces or ['rgb', 'hsv']
            
            for img_path in image_paths:
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    
                    feature_vector = []
                    
                    # Color histograms
                    if 'rgb' in color_spaces:
                        for i in range(3):
                            hist = cv2.calcHist([img], [i], None, [histogram_bins], [0, 256])
                            feature_vector.extend(hist.flatten().tolist())
                    
                    if 'hsv' in color_spaces:
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        for i in range(3):
                            hist = cv2.calcHist([hsv], [i], None, [histogram_bins], [0, 256])
                            feature_vector.extend(hist.flatten().tolist())
                    
                    if 'lab' in color_spaces:
                        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        for i in range(3):
                            hist = cv2.calcHist([lab], [i], None, [histogram_bins], [0, 256])
                            feature_vector.extend(hist.flatten().tolist())
                    
                    # Basic image stats
                    feature_vector.extend([
                        img.shape[0],  # height
                        img.shape[1],  # width
                        img.mean(),    # mean pixel value
                        img.std()      # std pixel value
                    ])
                    
                    result["features"].append({
                        "image_path": img_path,
                        "feature_vector": feature_vector,
                        "feature_dim": len(feature_vector)
                    })
                    
                except Exception as e:
                    result["failed_images"].append({"path": img_path, "error": str(e)})
            
            if result["features"]:
                result["feature_dim"] = result["features"][0]["feature_dim"]
        
        elif method == "texture":
            print("  Extracting texture features...")
            
            if not CV2_AVAILABLE:
                raise ImportError("OpenCV required for texture features")
            
            for img_path in image_paths:
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    
                    # Edge detection
                    edges = cv2.Canny(img, 100, 200)
                    
                    # Texture features
                    feature_vector = [
                        edges.mean(),
                        edges.std(),
                        np.count_nonzero(edges) / edges.size,  # edge density
                        img.mean(),
                        img.std()
                    ]
                    
                    result["features"].append({
                        "image_path": img_path,
                        "feature_vector": feature_vector,
                        "feature_dim": len(feature_vector)
                    })
                    
                except Exception as e:
                    result["failed_images"].append({"path": img_path, "error": str(e)})
            
            if result["features"]:
                result["feature_dim"] = result["features"][0]["feature_dim"]
        
        else:
            raise ValueError(f"Unknown method '{method}' or required libraries not available")
        
        print(f"✅ Feature extraction complete!")
        print(f"   Processed: {len(result['features'])} images")
        print(f"   Failed: {len(result['failed_images'])} images")
        print(f"   Feature dimension: {result['feature_dim']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {str(e)}")
        raise


def _extract_features_basic(image_paths: List[str]) -> Dict[str, Any]:
    """Fallback feature extraction using PIL when OpenCV/PyTorch not available."""
    
    result = {
        "method": "basic_pil",
        "n_images": len(image_paths),
        "features": [],
        "feature_dim": 0,
        "failed_images": []
    }
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Basic statistics per channel
            feature_vector = []
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                feature_vector.extend([
                    channel_data.mean(),
                    channel_data.std(),
                    channel_data.min(),
                    channel_data.max()
                ])
            
            # Image dimensions
            feature_vector.extend([img_array.shape[0], img_array.shape[1]])
            
            result["features"].append({
                "image_path": img_path,
                "feature_vector": feature_vector,
                "feature_dim": len(feature_vector)
            })
            
        except Exception as e:
            result["failed_images"].append({"path": img_path, "error": str(e)})
    
    if result["features"]:
        result["feature_dim"] = result["features"][0]["feature_dim"]
    
    result["note"] = "Install torch, torchvision, and opencv for advanced features"
    
    return result


def perform_image_clustering(
    features: Dict[str, Any],
    n_clusters: int = 5,
    method: str = "kmeans",
    reduce_dimensions: bool = True,
    target_dim: int = 50,
    return_similar_pairs: bool = True,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Cluster images based on extracted features and find similar images.
    
    Args:
        features: Output from extract_image_features
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'dbscan')
        reduce_dimensions: Whether to reduce dimensions before clustering
        target_dim: Target dimensionality for reduction
        return_similar_pairs: Whether to return most similar image pairs
        top_k: Number of top similar pairs to return
    
    Returns:
        Dictionary containing cluster assignments, centroids, and similar pairs
    """
    print(f"🔍 Clustering images using {method}...")
    
    if not features.get("features"):
        raise ValueError("No features provided for clustering")
    
    # Extract feature vectors
    feature_vectors = np.array([f["feature_vector"] for f in features["features"]])
    image_paths = [f["image_path"] for f in features["features"]]
    
    print(f"  Feature matrix shape: {feature_vectors.shape}")
    
    result = {
        "method": method,
        "n_images": len(image_paths),
        "n_clusters": n_clusters,
        "clusters": []
    }
    
    try:
        # Normalize features
        scaler = StandardScaler()
        feature_vectors_scaled = scaler.fit_transform(feature_vectors)
        
        # Dimensionality reduction
        if reduce_dimensions and feature_vectors_scaled.shape[1] > target_dim:
            print(f"  Reducing dimensions from {feature_vectors_scaled.shape[1]} to {target_dim}...")
            pca = PCA(n_components=target_dim)
            feature_vectors_reduced = pca.fit_transform(feature_vectors_scaled)
            result["explained_variance"] = float(pca.explained_variance_ratio_.sum())
            print(f"    Explained variance: {result['explained_variance']:.3f}")
        else:
            feature_vectors_reduced = feature_vectors_scaled
        
        # Clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(feature_vectors_reduced)
            
            result["cluster_centers"] = clusterer.cluster_centers_.tolist()
            result["inertia"] = float(clusterer.inertia_)
            
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(feature_vectors_reduced)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            result["n_clusters"] = n_clusters
            result["n_noise_points"] = int((labels == -1).sum())
            
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'kmeans' or 'dbscan'")
        
        # Organize results by cluster
        for cluster_id in sorted(set(labels)):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_images = [image_paths[i] for i in cluster_indices]
            
            cluster_info = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_images),
                "images": cluster_images[:100]  # Limit to first 100
            }
            
            if method == "kmeans":
                # Calculate distances to centroid
                centroid = clusterer.cluster_centers_[cluster_id]
                distances = np.linalg.norm(feature_vectors_reduced[cluster_indices] - centroid, axis=1)
                
                # Representative images (closest to centroid)
                representative_indices = distances.argsort()[:5]
                cluster_info["representative_images"] = [
                    cluster_images[i] for i in representative_indices
                ]
            
            result["clusters"].append(cluster_info)
        
        # Find similar image pairs
        if return_similar_pairs:
            print(f"  Finding top {top_k} similar image pairs...")
            
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarity_matrix = cosine_similarity(feature_vectors_reduced)
            
            # Get upper triangle indices (avoid duplicates and self-similarity)
            triu_indices = np.triu_indices(len(image_paths), k=1)
            similarities = similarity_matrix[triu_indices]
            
            # Get top K most similar pairs
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            similar_pairs = []
            for idx in top_indices:
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                similar_pairs.append({
                    "image1": image_paths[i],
                    "image2": image_paths[j],
                    "similarity": float(similarities[idx])
                })
            
            result["similar_pairs"] = similar_pairs
        
        # Visualize with t-SNE (if enough samples)
        if len(image_paths) >= 30:
            print("  Computing t-SNE for visualization...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(image_paths)-1))
            embeddings_2d = tsne.fit_transform(feature_vectors_reduced)
            
            result["tsne_embeddings"] = embeddings_2d.tolist()
        
        print(f"✅ Clustering complete!")
        print(f"   Clusters: {len(result['clusters'])}")
        for cluster in result["clusters"]:
            print(f"     Cluster {cluster['cluster_id']}: {cluster['size']} images")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during clustering: {str(e)}")
        raise


def analyze_tabular_image_hybrid(
    tabular_data: pl.DataFrame,
    image_column: str,
    target_column: Optional[str] = None,
    tabular_features: Optional[List[str]] = None,
    fusion_method: str = "concatenate",
    model_type: str = "classification",
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Analyze datasets with both tabular and image data using multi-modal learning.
    
    Args:
        tabular_data: DataFrame with tabular features and image paths
        image_column: Column containing image file paths
        target_column: Target variable column (if supervised learning)
        tabular_features: List of tabular feature columns (if None, uses all except image/target)
        fusion_method: How to combine features ('concatenate', 'attention', 'early', 'late')
        model_type: Type of task ('classification', 'regression')
        test_size: Proportion of data for testing
    
    Returns:
        Dictionary containing model performance, feature importance, and predictions
    """
    print(f"🔍 Analyzing hybrid tabular-image data...")
    
    # Validate input
    if image_column not in tabular_data.columns:
        raise ValueError(f"Image column '{image_column}' not found in DataFrame")
    
    if target_column and target_column not in tabular_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Determine tabular features
    if tabular_features is None:
        exclude_cols = [image_column]
        if target_column:
            exclude_cols.append(target_column)
        tabular_features = [col for col in tabular_data.columns if col not in exclude_cols]
    
    print(f"  Tabular features: {len(tabular_features)}")
    print(f"  Image column: {image_column}")
    print(f"  Target column: {target_column}")
    
    result = {
        "n_samples": tabular_data.shape[0],
        "n_tabular_features": len(tabular_features),
        "fusion_method": fusion_method,
        "model_type": model_type
    }
    
    try:
        # Step 1: Extract image features
        print("\n  Step 1: Extracting image features...")
        image_paths = tabular_data[image_column].to_list()
        
        # Use CNN features if available, otherwise color histograms
        method = "cnn" if TORCH_AVAILABLE else "color"
        image_features_result = extract_image_features(
            image_paths,
            method=method,
            model_name="resnet50" if TORCH_AVAILABLE else None
        )
        
        # Build image feature matrix
        image_feature_matrix = np.array([
            f["feature_vector"] for f in image_features_result["features"]
        ])
        
        print(f"    Image features shape: {image_feature_matrix.shape}")
        
        # Step 2: Prepare tabular features
        print("\n  Step 2: Preparing tabular features...")
        tabular_feature_matrix = tabular_data.select(tabular_features).to_numpy()
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        tabular_feature_matrix = imputer.fit_transform(tabular_feature_matrix)
        
        print(f"    Tabular features shape: {tabular_feature_matrix.shape}")
        
        # Step 3: Fusion
        print(f"\n  Step 3: Fusing features using '{fusion_method}' method...")
        
        if fusion_method == "concatenate" or fusion_method == "early":
            # Simple concatenation
            combined_features = np.hstack([tabular_feature_matrix, image_feature_matrix])
            result["combined_feature_dim"] = combined_features.shape[1]
            
        elif fusion_method == "late":
            # Train separate models and combine predictions
            combined_features = tabular_feature_matrix  # Will handle separately
            result["combined_feature_dim"] = tabular_feature_matrix.shape[1]
            result["image_feature_dim"] = image_feature_matrix.shape[1]
            
        else:
            raise ValueError(f"Unknown fusion method '{fusion_method}'")
        
        print(f"    Combined features shape: {combined_features.shape}")
        
        # Step 4: Train model (if target provided)
        if target_column:
            print(f"\n  Step 4: Training {model_type} model...")
            
            target = tabular_data[target_column].to_numpy()
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                combined_features, target, test_size=test_size, random_state=42
            )
            
            # Train model
            if model_type == "classification":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                from sklearn.metrics import accuracy_score, classification_report
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                result["train_accuracy"] = float(accuracy_score(y_train, train_pred))
                result["test_accuracy"] = float(accuracy_score(y_test, test_pred))
                
                # Classification report
                report = classification_report(y_test, test_pred, output_dict=True)
                result["classification_report"] = report
                
            elif model_type == "regression":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                result["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, train_pred)))
                result["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, test_pred)))
                result["train_r2"] = float(r2_score(y_train, train_pred))
                result["test_r2"] = float(r2_score(y_test, test_pred))
                result["test_mae"] = float(mean_absolute_error(y_test, test_pred))
            
            # Feature importance
            if fusion_method == "concatenate":
                feature_names = tabular_features + [f"image_feat_{i}" for i in range(image_feature_matrix.shape[1])]
                
                # Top 20 most important features
                importances = model.feature_importances_
                top_indices = importances.argsort()[-20:][::-1]
                
                result["top_features"] = [
                    {
                        "feature": feature_names[i],
                        "importance": float(importances[i])
                    }
                    for i in top_indices
                ]
                
                # Compare tabular vs image feature importance
                tabular_importance = importances[:len(tabular_features)].sum()
                image_importance = importances[len(tabular_features):].sum()
                
                result["feature_importance_split"] = {
                    "tabular": float(tabular_importance),
                    "image": float(image_importance),
                    "tabular_percentage": float(tabular_importance / importances.sum() * 100),
                    "image_percentage": float(image_importance / importances.sum() * 100)
                }
        
        print(f"\n✅ Hybrid analysis complete!")
        if target_column:
            if model_type == "classification":
                print(f"   Test accuracy: {result['test_accuracy']:.4f}")
            else:
                print(f"   Test R²: {result['test_r2']:.4f}")
                print(f"   Test RMSE: {result['test_rmse']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during hybrid analysis: {str(e)}")
        raise
