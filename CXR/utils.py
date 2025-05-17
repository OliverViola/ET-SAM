import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional

def compute_cosine_distance(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """
    Compute cosine distance between two feature vectors.
    Args:
        z1, z2: Feature vectors from SAM2 encoder
    Returns:
        Cosine distance between z1 and z2
    """
    z1_norm = torch.nn.functional.normalize(z1.flatten(), dim=0)
    z2_norm = torch.nn.functional.normalize(z2.flatten(), dim=0)
    return torch.dot(z1_norm, z2_norm).item()

def compute_similarity(features: List[torch.Tensor], idx: int) -> float:
    """
    Compute similarity score for a sample according to Equation (2).
    Args:
        features: List of feature vectors from SAM2 encoder
        idx: Index of the sample to compute similarity for
    Returns:
        Similarity score for the sample
    """
    N = len(features)
    similarity_sum = 0.0
    
    for j in range(N):
        if j != idx:
            similarity_sum += compute_cosine_distance(features[idx], features[j])
    
    return (1.0 / (N - 1)) * similarity_sum

def find_most_similar_sample(features: List[torch.Tensor]) -> int:
    """
    Find the sample with highest similarity according to Equation (3).
    Args:
        features: List of feature vectors from SAM2 encoder
    Returns:
        Index of the most similar sample
    """
    similarities = [compute_similarity(features, i) for i in range(len(features))]
    return int(np.argmax(similarities))

def k_center_greedy(features: List[torch.Tensor], k: int, first_center_idx: Optional[int] = None) -> List[int]:
    """
    Implement K-Center-Greedy algorithm for selecting diverse samples.
    Args:
        features: List of feature vectors from SAM2 encoder
        k: Number of centers to select
        first_center_idx: Index of first center (if None, use most similar sample)
    Returns:
        Indices of selected centers
    """
    n_samples = len(features)
    if first_center_idx is None:
        first_center_idx = find_most_similar_sample(features)
    
    centers = [first_center_idx]
    distances = torch.zeros(n_samples)
    
    # Convert features to tensor for faster computation
    features_tensor = torch.stack([f.flatten() for f in features])
    features_tensor = torch.nn.functional.normalize(features_tensor, dim=1)
    
    while len(centers) < k:
        # Compute distances to nearest center for all points
        center_features = features_tensor[centers[-1]].unsqueeze(0)
        dist_to_center = 1 - torch.mm(features_tensor, center_features.t()).squeeze()
        
        # Update minimum distances
        distances = torch.minimum(distances, dist_to_center)
        
        # Select the point with maximum distance as the next center
        next_center = int(torch.argmax(distances))
        centers.append(next_center)
    
    return centers

def select_few_shot_samples(features: List[torch.Tensor], k: int) -> Tuple[List[int], List[List[int]]]:
    """
    Select few-shot training samples using K-Centroid algorithm with K-Center-Greedy initialization.
    Args:
        features: List of feature vectors from SAM2 encoder
        k: Number of samples to select
    Returns:
        Tuple of (selected sample indices, list of query sample indices for each support sample)
    """
    # Find most similar sample as first center
    s1_idx = find_most_similar_sample(features)
    
    # Use K-Center-Greedy to find initial centroids
    initial_centers = k_center_greedy(features, k, s1_idx)
    
    # Convert features to numpy for K-Means
    features_np = torch.stack([f.flatten() for f in features]).cpu().numpy()
    
    # Run K-Means with initial centers
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
    kmeans.fit(features_np)
    
    # Find closest samples to centroids as support samples
    support_samples = []
    query_groups = [[] for _ in range(k)]
    
    for cluster_idx in range(k):
        cluster_points = np.where(kmeans.labels_ == cluster_idx)[0]
        
        # Find point closest to centroid as support sample
        distances_to_centroid = np.linalg.norm(
            features_np[cluster_points] - kmeans.cluster_centers_[cluster_idx].reshape(1, -1),
            axis=1
        )
        support_idx = cluster_points[np.argmin(distances_to_centroid)]
        support_samples.append(int(support_idx))
        
        # Assign remaining points as query samples
        query_samples = [int(idx) for idx in cluster_points if idx != support_idx]
        query_groups[cluster_idx] = query_samples
    
    return support_samples, query_groups


