# Interactive 3D Dashboard for Memory Yield Analysis
# Author: Kahsay Kiross Meresa
# Email: kahsaykirossmtc@gmail.com

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.express as px
import dash
from dash import dcc, html

# ------------------------------
# 1. Load Dataset
# ------------------------------
def load_memory_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

# ------------------------------
# 2. Tensor Decomposition (CP)
# ------------------------------
def tensor_decomposition(X, rank=10):
    tensor = X.reshape(X.shape[0], X.shape[1], 1)
    factors = parafac(tensor, rank=rank)
    X_reduced = tl.kruskal_to_tensor(factors).reshape(X.shape[0], -1)
    return X_reduced

# ------------------------------
# 3. Meta-Learning Model
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
def adaptive_sampling(model, X_pool, num_samples=500):
    Y_pred = model.predict(X_pool, verbose=0).flatten()
    weights = Y_pred / np.sum(Y_pred)
    sampled_indices = np.random.choice(len(X_pool), size=num_samples, p=weights)
    return X_pool[sampled_indices], Y_pred[sampled_indices]

# ------------------------------
# 5. LVF2 Gaussian Mixture Model
# ------------------------------
def lvf2_gmm(X, n_components=3):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    return gmm

# ------------------------------
# 6. Prepare Data for 3D Visualization
# ------------------------------
def prepare_3d_data(X, Y, rare_percentile=5):
    # Reduce to 3D using PCA for visualization
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)
    threshold = np.percentile(Y, rare_percentile)
    rare_mask = Y <= threshold
    return X_3d, Y, rare_mask, threshold

# ------------------------------
# 7. Build Dash App
# ------------------------------
def build_dash_app(X_3d, Y, rare_mask, threshold):
    app = dash.Dash(__name__)
    
    fig = px.scatter_3d(
        x=X_3d[:,0], y=X_3d[:,1], z=X_3d[:,2],
        color=rare_mask.map({True:'Rare Event', False:'Normal'}),
        size=Y*50, # scale for visualization
        hover_data={'Yield': Y, 'Rare Event': rare_mask},
        color_discrete_map={'Rare Event':'red','Normal':'blue'},
        title=f"3D Multi-Failure Region & Rare Events (Threshold: {threshold:.4f})"
    )
    
    app.layout = html.Div([
        html.H1("High-Dimensional Memory Yield Analysis Dashboard"),
        dcc.Graph(figure=fig),
        html.P(f"Number of Rare Events: {rare_mask.sum()} / Total Samples: {len(Y)}")
    ])
    
    return app

# ------------------------------
# 8. Main Workflow
# ------------------------------
if __name__ == "__main__":
    # Load real memory dataset
    X, Y = load_memory_dataset("real_memory_dataset.csv")
    
    # Tensor decomposition
    X_reduced = tensor_decomposition(X, rank=10)
    
    # Train meta-learning model
    X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=42)
    meta_model = build_meta_model(input_dim=X_train.shape[1])
    meta_model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_split=0.1, verbose=0)
    
    # Adaptive importance sampling
    X_sampled, Y_sampled = adaptive_sampling(meta_model, X_test, num_samples=1000)
    
    # Prepare 3D data
    X_3d, Y_vis, rare_mask, threshold = prepare_3d_data(X_sampled, Y_sampled)
    
    # Launch interactive dashboard
    app = build_dash_app(X_3d, Y_vis, pd.Series(rare_mask), threshold)
    app.run_server(debug=True)
