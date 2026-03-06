# AI-Driven High-Dimensional Memory Yield Analysis
# Author: Kahsay Kiross Meresa
# Email: kahsaykirossmtc@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------------------
# 1. Generate High-Dimensional Memory Dataset
# ------------------------------
def generate_memory_dataset(num_samples=10000, num_params=50):
    """
    Simulate high-dimensional process variations for SRAM/memory circuits
    """
    mean = np.zeros(num_params)
    cov = np.identity(num_params) * 0.05  # process variation
    X = np.random.multivariate_normal(mean, cov, size=num_samples)
    # Yield function: lower sum of squares = higher yield
    Y = np.exp(-np.sum(X**2, axis=1))
    return X, Y

# ------------------------------
# 2. Low-Rank Tensor Approximation using TruncatedSVD
# ------------------------------
def low_rank_tensor(X, n_components=10):
    """
    Reduce dimensionality of high-dimensional memory parameters
    """
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd

# ------------------------------
# 3. Meta-Learning Model using Neural Network
# ------------------------------
def build_meta_model(input_dim):
    """
    Neural network to predict memory yield
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# ------------------------------
# 4. Adaptive Importance Sampling
# ------------------------------
def adaptive_importance_sampling(model, X_pool, num_samples=500):
    """
    Sample high-yield or low-yield regions efficiently
    """
    Y_pred = model.predict(X_pool, verbose=0).flatten()
    weights = Y_pred / np.sum(Y_pred)
    sampled_indices = np.random.choice(len(X_pool), size=num_samples, p=weights)
    return X_pool[sampled_indices], Y_pred[sampled_indices]

# ------------------------------
# 5. Tail Distribution Analysis for Rare Failures
# ------------------------------
def tail_distribution(Y, percentile=5):
    """
    Identify rare low-yield events (e.g., failures)
    """
    threshold = np.percentile(Y, percentile)
    rare_events = Y[Y <= threshold]
    return rare_events, threshold

# ------------------------------
# 6. Visualization Functions
# ------------------------------
def plot_yield_distribution(Y, rare_events, threshold):
    plt.figure(figsize=(8,5))
    plt.hist(Y, bins=50, alpha=0.7, label='Yield Distribution')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Rare-event threshold ({threshold:.4f})')
    plt.scatter(rare_events, np.zeros_like(rare_events), color='red', marker='x', label='Rare events')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.title('Memory Yield Distribution and Rare Events')
    plt.legend()
    plt.show()

# ------------------------------
# 7. Main Workflow
# ------------------------------
if __name__ == "__main__":
    # Generate dataset
    X, Y = generate_memory_dataset(num_samples=10000, num_params=50)
    
    # Low-rank approximation
    X_reduced, svd_model = low_rank_tensor(X, n_components=10)
    
    # Train meta-learning model
    X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=42)
    meta_model = build_meta_model(input_dim=X_train.shape[1])
    meta_model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_split=0.1, verbose=0)
    
    # Adaptive importance sampling
    X_sampled, Y_sampled = adaptive_importance_sampling(meta_model, X_test, num_samples=500)
    
    # Tail distribution analysis
    rare_events, threshold = tail_distribution(Y_sampled, percentile=5)
    
    # Visualization
    plot_yield_distribution(Y_sampled, rare_events, threshold)
    
    # Summary
    print(f"Original dataset shape: {X.shape}")
    print(f"Reduced dataset shape: {X_reduced.shape}")
    print(f"Number of rare events detected: {len(rare_events)}")
    print(f"Rare-event threshold (bottom 5% yield): {threshold:.4f}")
