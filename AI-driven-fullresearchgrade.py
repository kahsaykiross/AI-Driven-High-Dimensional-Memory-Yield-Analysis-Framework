# Full Research-Grade AI-Driven Memory Yield Analysis
# Author: Kahsay Kiross Meresa
# Email: kahsaykirossmtc@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------------------
# 1. Load Real SRAM / AI Memory Dataset
# ------------------------------
def load_memory_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

# ------------------------------
# 2. Tensor Decomposition (CP/Tucker)
# ------------------------------
def tensor_decomposition(X, method='cp', rank=10):
    tensor = X.reshape(X.shape[0], X.shape[1], 1)
    if method.lower() == 'cp':
        factors = parafac(tensor, rank=rank)
        X_reduced = tl.kruskal_to_tensor(factors).reshape(X.shape[0], -1)
    elif method.lower() == 'tucker':
        core, factors = tucker(tensor, ranks=[rank, rank, 1])
        X_reduced = tl.tucker_to_tensor((core, factors)).reshape(X.shape[0], -1)
    else:
        raise ValueError("Method must be 'cp' or 'tucker'")
    return X_reduced

# ------------------------------
# 3. Meta-Learning Neural Network
# ------------------------------
def build_meta_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# ------------------------------
# 4. Adaptive Importance Sampling
# ------------------------------
def adaptive_importance_sampling(model, X_pool, num_samples=500):
    Y_pred = model.predict(X_pool, verbose=0).flatten()
    weights = Y_pred / np.sum(Y_pred)
    sampled_indices = np.random.choice(len(X_pool), size=num_samples, p=weights)
    return X_pool[sampled_indices], Y_pred[sampled_indices]

# ------------------------------
# 5. Tail Distribution / Rare-Event Analysis
# ------------------------------
def tail_distribution(Y, percentile=5):
    threshold = np.percentile(Y, percentile)
    rare_events = Y[Y <= threshold]
    return rare_events, threshold

# ------------------------------
# 6. LVF2 Gaussian Mixture Model (Timing Binning)
# ------------------------------
def lvf2_gmm(X, n_components=3):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    return gmm

# ------------------------------
# 7. Multi-Failure Region Probability Map
# ------------------------------
def plot_failure_heatmap(X, Y, threshold):
    """
    Visualize multi-failure regions by projecting first two principal components
    """
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=(Y<=threshold), palette={True:'red', False:'blue'}, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Multi-Failure Region Heatmap (Red = Rare Events)")
    plt.show()

# ------------------------------
# 8. Speed Binning Plot from GMM
# ------------------------------
def plot_speed_bins(gmm_model):
    plt.figure(figsize=(7,5))
    for i, mean in enumerate(gmm_model.means_):
        plt.axvline(mean[0], linestyle='--', label=f'Bin {i+1} Mean: {mean[0]:.3f}')
    plt.xlabel("Yield / Speed Metric")
    plt.ylabel("Density")
    plt.title("LVF2 Gaussian Mixture Speed Bins")
    plt.legend()
    plt.show()

# ------------------------------
# 9. Rare-Event Heatmap (Top 2 PCs)
# ------------------------------
def rare_event_heatmap(X, rare_events_mask):
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=rare_events_mask, palette={True:'red', False:'green'}, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Rare Event Heatmap (Red = Rare / Critical Failures)")
    plt.show()

# ------------------------------
# 10. System-Level Benchmark Evaluation
# ------------------------------
def benchmark_evaluation(model, benchmark_X, benchmark_Y):
    Y_pred = model.predict(benchmark_X, verbose=0).flatten()
    mse = np.mean((Y_pred - benchmark_Y)**2)
    print(f"[Benchmark] System-level MSE: {mse:.6f}")
    return Y_pred, mse

# ------------------------------
# 11. Main Workflow
# ------------------------------
if __name__ == "__main__":
    # Load real memory dataset
    X, Y = load_memory_dataset("real_memory_dataset.csv")
    
    # Tensor decomposition (CP or Tucker)
    X_reduced = tensor_decomposition(X, method='cp', rank=10)
    
    # Split train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=42)
    
    # Train meta-learning model
    meta_model = build_meta_model(input_dim=X_train.shape[1])
    meta_model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_split=0.1, verbose=0)
    
    # Adaptive importance sampling
    X_sampled, Y_sampled = adaptive_importance_sampling(meta_model, X_test, num_samples=500)
    
    # Tail distribution analysis
    rare_events, threshold = tail_distribution(Y_sampled, percentile=5)
    
    # Multi-failure region heatmap
    plot_failure_heatmap(X_sampled, Y_sampled, threshold)
    
    # LVF2 Gaussian Mixture for timing-speculation
    gmm_model = lvf2_gmm(X_sampled, n_components=3)
    plot_speed_bins(gmm_model)
    
    # Rare-event heatmap
    rare_mask = Y_sampled <= threshold
    rare_event_heatmap(X_sampled, rare_mask)
    
    # System-level benchmark (replace with actual benchmark dataset)
    benchmark_X, benchmark_Y = load_memory_dataset("fpga_ai_benchmark.csv")
    Y_pred_benchmark, mse_benchmark = benchmark_evaluation(meta_model, benchmark_X, benchmark_Y)
    
    # Summary
    print(f"Original dataset shape: {X.shape}")
    print(f"Reduced dataset shape: {X_reduced.shape}")
    print(f"Number of rare events: {len(rare_events)}")
    print(f"Rare-event threshold (bottom 5% yield): {threshold:.4f}")
