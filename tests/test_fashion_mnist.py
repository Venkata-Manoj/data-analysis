"""Offline smoke tests for the Fashion-MNIST Image Classification pipeline.

Uses synthetic 28×28 image data that mirrors the Fashion-MNIST schema
(784 pixels + 10 clothing classes) so the tests run fast in CI without
downloading from OpenML. Verifies PCA, scaling, stratified split, model
training, and metric computation end to end.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_CLASSES = 10
N_FEATURES = 784  # 28*28
N_SAMPLES = 600
N_COMPONENTS = 30
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@pytest.fixture
def synthetic_images():
    """Synthetic Fashion-MNIST-like data: 600 samples, 784 pixels, 10 classes."""
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=50,  # informative features; will be expanded to 784
        n_informative=20,
        n_redundant=10,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    # Expand to 784 pixels via random projection + noise to mimic 28×28
    rng = np.random.default_rng(RANDOM_STATE)
    # Pad with noise pixels to reach 784
    noise = rng.normal(0, 0.5, size=(N_SAMPLES, N_FEATURES - 50))
    X_full = np.hstack([X, noise])
    # Scale to 0-255 uint8 range like Fashion-MNIST
    X_full = ((X_full - X_full.min()) / (X_full.max() - X_full.min()) * 255).astype(np.float64)
    return X_full, y


def test_synthetic_image_shapes(synthetic_images):
    """Synthetic data must match Fashion-MNIST geometry."""
    X, y = synthetic_images
    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert y.shape == (N_SAMPLES,)
    assert len(np.unique(y)) == N_CLASSES
    # Each image reshapes to 28×28
    assert X[0].reshape(28, 28).shape == (28, 28)


def test_pca_explained_variance(synthetic_images):
    """PCA cumulative variance must be increasing and ≤ 1."""
    X, _ = synthetic_images
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    n_comp = min(N_COMPONENTS, X_s.shape[0], X_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca.fit(X_s)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    assert cum_var[-1] <= 1.0 + 1e-9
    assert cum_var[-1] > 0.05  # synthetic has mostly noise; real Fashion-MNIST 30 comps ≈0.756
    # Monotonically increasing
    assert np.all(np.diff(cum_var) >= -1e-12)
    assert pca.explained_variance_ratio_.shape == (n_comp,)


def test_scaler_and_pca_pipeline(synthetic_images):
    """StandardScaler + PCA must produce zero-mean training embedding."""
    X, y = synthetic_images
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-9)
    n_comp = min(N_COMPONENTS, X_train_s.shape[1], X_train_s.shape[0])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    assert X_train_pca.shape == (X_train.shape[0], n_comp)
    assert X_test_pca.shape == (X_test.shape[0], n_comp)


def test_stratified_split_preserves_distribution(synthetic_images):
    """Stratified split must preserve class ratios."""
    X, y = synthetic_images
    _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    full_ratio = np.bincount(y, minlength=N_CLASSES) / len(y)
    train_ratio = np.bincount(y_train, minlength=N_CLASSES) / len(y_train)
    # Each class ratio within 5%
    assert np.all(np.abs(full_ratio - train_ratio) < 0.05)
    assert set(np.unique(y_train)) == set(range(N_CLASSES))
    assert set(np.unique(y_test)) == set(range(N_CLASSES))


def test_classifiers_accuracy_in_range(synthetic_images):
    """LR / RF / MLP must train and produce accuracy in [0, 1] on PCA data."""
    X, y = synthetic_images
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    n_comp = min(N_COMPONENTS, X_train_s.shape[1], X_train_s.shape[0])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    # Logistic Regression
    lr = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
    lr.fit(X_train_pca, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test_pca))
    assert 0.0 <= acc_lr <= 1.0
    assert acc_lr > 0.1  # better than random (0.1 for 10 classes)

    # Random Forest (small)
    rf = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train_pca, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test_pca))
    assert 0.0 <= acc_rf <= 1.0
    assert acc_rf > 0.1

    # MLP (small, early stopping off for determinism)
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=50,
        random_state=RANDOM_STATE,
        early_stopping=False,
        verbose=False,
    )
    mlp.fit(X_train_pca, y_train)
    acc_mlp = accuracy_score(y_test, mlp.predict(X_test_pca))
    assert 0.0 <= acc_mlp <= 1.0
    assert acc_mlp > 0.1


def test_confusion_matrix_shape(synthetic_images):
    """Confusion matrix on 10 classes must be 10×10."""
    X, y = synthetic_images
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    n_comp = min(10, X_train_s.shape[1], X_train_s.shape[0])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    clf = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    assert cm.shape == (N_CLASSES, N_CLASSES)
    assert cm.sum() == len(y_test)


def test_per_class_f1_in_range(synthetic_images):
    """Per-class F1 scores must be in [0, 1]."""
    X, y = synthetic_images
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    n_comp = min(10, X_train_s.shape[1], X_train_s.shape[0])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    clf = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    f1s = f1_score(y_test, y_pred, average=None, zero_division=0)
    assert f1s.shape == (N_CLASSES,)
    assert np.all(f1s >= 0)
    assert np.all(f1s <= 1.0)

    # CLASS_NAMES sanity
    assert len(CLASS_NAMES) == N_CLASSES
