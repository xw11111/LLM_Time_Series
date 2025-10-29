import gc
import time
import pickle
import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from typing import Optional

# Check GPU availability
def check_gpu_setup():
    """Check what GPU libraries are available"""
    gpu_info = {
        'torch_cuda': False,
        'cuml': False,
        'device_count': 0,
        'gpu_names': []
    }
    
    # Check PyTorch CUDA
    try:
        import torch
        gpu_info['torch_cuda'] = torch.cuda.is_available()
        if gpu_info['torch_cuda']:
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(gpu_info['device_count'])]
    except ImportError:
        pass
    
    # Check cuML
    try:
        import cuml
        from cuml.cluster import KMeans as cuKMeans
        gpu_info['cuml'] = True
    except (ImportError, RuntimeError) as e:
        print(f"cuML not available: {e}")
        gpu_info['cuml'] = False
    
    return gpu_info

class PyTorchKMeans:
    """GPU-accelerated K-means using PyTorch (more reliable than cuML on SageMaker)"""
    
    def __init__(self, n_clusters, max_iter=200, tol=1e-4, batch_size=32768, random_state=42, verbose=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.centroids = None
        self.labels_ = None
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print(f"PyTorchKMeans using device: {self.device}")
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def _init_centroids_plus_plus(self, X):
        """K-means++ initialization"""
        n_samples, n_features = X.shape
        centroids = torch.empty((self.n_clusters, n_features), dtype=X.dtype, device=X.device)
        
        # Choose first centroid randomly
        first_idx = torch.randint(0, n_samples, (1,), device=X.device)
        centroids[0] = X[first_idx]
        
        for c in range(1, self.n_clusters):
            # Calculate squared distances to nearest existing centroid
            distances = torch.cdist(X, centroids[:c]).min(dim=1)[0] ** 2
            
            # Convert to probabilities
            probs = distances / distances.sum()
            
            # Sample next centroid
            cumulative_probs = torch.cumsum(probs, dim=0)
            r = torch.rand(1, device=X.device)
            next_idx = torch.searchsorted(cumulative_probs, r)
            next_idx = torch.clamp(next_idx, 0, n_samples - 1)
            centroids[c] = X[next_idx]
            
        return centroids
    
    def _assign_clusters_batched(self, X, centroids):
        """Assign points to clusters in batches to manage GPU memory"""
        n_samples = X.shape[0]
        labels = torch.empty(n_samples, dtype=torch.long, device=X.device)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = X[start_idx:end_idx]
            
            # Calculate squared distances to all centroids
            distances = torch.cdist(batch, centroids) ** 2
            batch_labels = torch.argmin(distances, dim=1)
            labels[start_idx:end_idx] = batch_labels
            
        return labels
    
    def fit(self, X):
        """Fit K-means to data"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"Fitting K-means: {n_samples} samples, {n_features} features, {self.n_clusters} clusters")
        
        # Initialize centroids
        centroids = self._init_centroids_plus_plus(X)
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Assign points to clusters
            labels = self._assign_clusters_batched(X, centroids)
            
            # Update centroids
            new_centroids = torch.empty_like(centroids)
            total_loss = 0.0
            
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    cluster_points = X[mask]
                    new_centroids[k] = cluster_points.mean(dim=0)
                    # Calculate within-cluster sum of squares
                    total_loss += ((cluster_points - new_centroids[k]) ** 2).sum().item()
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]
            
            # Check for convergence
            loss_change = abs(prev_loss - total_loss)
            centroid_shift = torch.norm(new_centroids - centroids).item()
            
            if self.verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: loss={total_loss:.2f}, centroid_shift={centroid_shift:.6f}")
            
            if centroid_shift < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
            centroids = new_centroids
            prev_loss = total_loss
        
        self.centroids = centroids
        self.labels_ = labels
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        return self._assign_clusters_batched(X, self.centroids)
    
    def get_centroids_numpy(self):
        """Get centroids as numpy array"""
        return self.centroids.cpu().numpy()
    
    def get_labels_numpy(self):
        """Get labels as numpy array"""
        return self.labels_.cpu().numpy()

def clustering_pytorch(wave, num_clusters, model_filename, sample_silhouette=20000):
    """PyTorch GPU implementation (most reliable)"""
    start_time = time.time()
    
    print(f"Using PyTorch GPU clustering: {len(wave)} samples, {wave.shape[1]}D")
    
    # Use PyTorch K-means
    kmeans = PyTorchKMeans(
        n_clusters=num_clusters,
        max_iter=200,
        tol=1e-4,
        batch_size=16384,  # Conservative batch size for stability
        random_state=42,
        verbose=True
    )
    
    # Fit the model
    kmeans.fit(wave)
    labels = kmeans.get_labels_numpy()
    centers = kmeans.get_centroids_numpy()
    
    fit_time = time.time() - start_time
    print(f"PyTorch clustering completed in {fit_time:.2f} seconds")
    
    # Calculate silhouette score (sampled for large datasets)
    if sample_silhouette and len(wave) > sample_silhouette:
        sample_indices = np.random.choice(len(wave), sample_silhouette, replace=False)
        sample_data = wave[sample_indices]
        sample_labels = labels[sample_indices]
        silhouette_avg = silhouette_score(sample_data, sample_labels)
        print(f"Silhouette Score (sampled): {silhouette_avg:.4f}")
    else:
        silhouette_avg = silhouette_score(wave, labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Create sklearn-compatible model
    sklearn_model = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    sklearn_model.cluster_centers_ = centers
    sklearn_model.labels_ = labels
    sklearn_model.n_features_in_ = centers.shape[1]
    
    # Save the model
    with open(model_filename, 'wb') as f:
        pickle.dump({'model': sklearn_model}, f)
    print(f"Model saved to {model_filename}")
    
    return labels

def clustering_cuml_safe(wave, num_clusters, model_filename, sample_silhouette=20000):
    """Safe cuML implementation with better error handling"""
    try:
        import cudf
        from cuml.cluster import KMeans as cuKMeans
        from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
        
        start_time = time.time()
        print(f"Attempting cuML clustering: {len(wave)} samples")
        
        # Clear GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Convert to cuDF with error handling
        X_np = np.asarray(wave, dtype=np.float32)
        X_gdf = cudf.DataFrame(X_np)
        
        # Create and fit cuML KMeans
        km = cuKMeans(
            n_clusters=num_clusters,
            init="scalable-k-means++",
            max_iter=100,
            tol=1e-4,
            random_state=42,
            verbose=0,
        )
        
        km.fit(X_gdf)
        labels = km.labels_.to_array()
        centers = km.cluster_centers_.to_pandas().values.astype(np.float32)
        
        # Calculate silhouette score
        if sample_silhouette and len(X_gdf) > sample_silhouette:
            idx = np.random.choice(len(X_gdf), size=sample_silhouette, replace=False)
            sil = float(cu_silhouette_score(X_gdf.iloc[idx], km.labels_.iloc[idx]))
        else:
            sil = float(cu_silhouette_score(X_gdf, km.labels_))
        
        fit_time = time.time() - start_time
        print(f"cuML clustering completed in {fit_time:.2f} seconds")
        print(f"Silhouette Score: {sil:.4f}")
        
        # Create sklearn-compatible model
        sklearn_model = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
        sklearn_model.cluster_centers_ = centers
        sklearn_model.labels_ = labels
        sklearn_model.n_features_in_ = centers.shape[1]
        
        # Save the model
        with open(model_filename, 'wb') as f:
            pickle.dump({'model': sklearn_model}, f)
        print(f"Model saved to {model_filename}")
        
        return labels
        
    except Exception as e:
        print(f"cuML failed with error: {e}")
        raise

def clustering(wave: np.ndarray, num_clusters: int, model_filename: str, sample_silhouette: int = 20000):
    """
    Main clustering function with automatic fallback hierarchy:
    1. Try cuML (fastest if working)
    2. Fall back to PyTorch GPU (most reliable GPU option)
    """
    
    # Check what's available
    gpu_info = check_gpu_setup()
    print(f"GPU Setup: {gpu_info}")
    
    # Clear memory
    gc.collect()
    if gpu_info['torch_cuda']:
        torch.cuda.empty_cache()
    
    # Try methods in order of preference
    methods = []
    
    if gpu_info['cuml']:
        methods.append(("cuML", clustering_cuml_safe))
    
    if gpu_info['torch_cuda']:
        methods.append(("PyTorch GPU", clustering_pytorch))
    
    for method_name, method_func in methods:
        try:
            print(f"\n=== Trying {method_name} ===")
            return method_func(wave, num_clusters, model_filename, sample_silhouette)
        except Exception as e:
            print(f"{method_name} failed: {e}")
            if method_name == methods[-1][0]:  # Last method
                print("All methods failed!")
                raise
            else:
                print(f"Falling back to next method...")
                continue

def perform_clustering(data: np.ndarray, num_clusters: int, model_filename: str, wave_type: str):
    """Main interface function"""
    if data is None or len(data) == 0:
        print(f"No data to cluster for {wave_type}")
        return np.array([])
    
    print(f"\n{'='*50}")
    print(f"Clustering {wave_type}: {len(data)} samples, {data.shape[1]} dimensions")
    print(f"Target clusters: {num_clusters}")
    print(f"{'='*50}")
    
    return clustering(data, num_clusters, model_filename)