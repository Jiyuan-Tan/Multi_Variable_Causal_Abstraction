import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def learn_hyperplane(
    df: pd.DataFrame,
    label_col: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
    normalize: bool = True
) -> Tuple[LogisticRegression, StandardScaler, dict]:
    """
    Learn a hyperplane to separate binary labeled data (0 or 1).
    
    This function uses Logistic Regression with L2 regularization to find an optimal
    linear decision boundary (hyperplane) that separates the two classes. It's efficient for large datasets (~4000 samples) and provides interpretable results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features and labels
    label_col : str, default='label'
        Name of the column containing binary labels (0 or 1)
    test_size : float, default=0.2
        Proportion of data to use for testing (between 0 and 1)
    random_state : int, default=42
        Random seed for reproducibility
    C : float, default=1.0
        Inverse of regularization strength. Smaller values = stronger regularization
    max_iter : int, default=1000
        Maximum number of iterations for the solver
    normalize : bool, default=True
        Whether to standardize features (recommended for numerical stability)
    
    Returns:
    --------
    model : LogisticRegression
        Trained logistic regression model
    scaler : StandardScaler or None
        Feature scaler (None if normalize=False)
    results : dict
        Dictionary containing:
        - 'train_accuracy': Training set accuracy
        - 'test_accuracy': Test set accuracy
        - 'coefficients': Hyperplane coefficients (weights)
        - 'intercept': Hyperplane intercept (bias)
        - 'n_features': Number of features
        - 'n_samples': Number of samples
        - 'class_distribution': Count of each class
    
    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5, 6],
    ...     'feature2': [2, 3, 4, 5, 6, 7],
    ...     'label': [0, 0, 0, 1, 1, 1]
    ... })
    >>> model, scaler, results = learn_hyperplane(df)
    >>> print(f"Test accuracy: {results['test_accuracy']:.3f}")
    >>> 
    >>> # Make predictions on new data
    >>> new_data = pd.DataFrame({'feature1': [2.5], 'feature2': [3.5]})
    >>> if scaler is not None:
    ...     new_data_scaled = scaler.transform(new_data)
    ... else:
    ...     new_data_scaled = new_data
    >>> prediction = model.predict(new_data_scaled)
    """
    
    # Validate input
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")
    
    # Check for binary labels
    unique_labels = df[label_col].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1. Found: {unique_labels}")
    
    # Separate features and labels
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    
    # Get dataset info
    n_samples, n_features = X.shape
    class_counts = pd.Series(y).value_counts().to_dict()
    
    print(f"Dataset info:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Class distribution: {class_counts}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Train logistic regression model
    print(f"\nTraining logistic regression (C={C}, max_iter={max_iter})...")
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',  # Efficient for medium datasets
        penalty='l2'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    
    # Extract hyperplane parameters
    coefficients = model.coef_[0]  # Shape: (n_features,)
    intercept = model.intercept_[0]  # Scalar
    
    print(f"\nHyperplane equation:")
    print(f"  Intercept (bias): {intercept:.4f}")
    print(f"  Coefficients (weights): {coefficients}")
    
    # Prepare results dictionary
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'coefficients': coefficients,
        'intercept': intercept,
        'n_features': n_features,
        'n_samples': n_samples,
        'class_distribution': class_counts,
        'feature_names': df.drop(columns=[label_col]).columns.tolist()
    }
    
    return model, scaler, results


def predict_with_hyperplane(
    model: LogisticRegression,
    X_new: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Make predictions on new data using the learned hyperplane.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model from learn_hyperplane()
    X_new : pd.DataFrame
        New data to predict (features only, no label column)
    scaler : StandardScaler or None
        Scaler used during training (if any)
    
    Returns:
    --------
    predictions : np.ndarray
        Predicted labels (0 or 1)
    """
    X = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
    
    if scaler is not None:
        X = scaler.transform(X)
    
    return model.predict(X)


def get_decision_scores(
    model: LogisticRegression,
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Get the decision function scores (distance from hyperplane).
    
    Positive scores indicate class 1, negative scores indicate class 0.
    The magnitude indicates confidence.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model from learn_hyperplane()
    X : pd.DataFrame
        Data to score (features only)
    scaler : StandardScaler or None
        Scaler used during training (if any)
    
    Returns:
    --------
    scores : np.ndarray
        Decision function scores
    """
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    if scaler is not None:
        X_array = scaler.transform(X_array)
    
    return model.decision_function(X_array)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Hyperplane Classifier - Example Usage")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 4000
    n_features = 10
    
    # Create two clusters
    X_class0 = np.random.randn(n_samples // 2, n_features) - 1
    X_class1 = np.random.randn(n_samples // 2, n_features) + 1
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Create dataframe
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nGenerated synthetic dataset with {n_samples} samples and {n_features} features")
    
    # Learn hyperplane
    model, scaler, results = learn_hyperplane(df, label_col='label')
    
    print("\n" + "=" * 60)
    print("Example: Making predictions on new data")
    print("=" * 60)
    
    # Test on a few samples
    test_samples = df.drop(columns=['label']).head(5)
    predictions = predict_with_hyperplane(model, test_samples, scaler)
    scores = get_decision_scores(model, test_samples, scaler)
    
    print("\nPredictions on first 5 samples:")
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        print(f"  Sample {i}: Predicted class = {pred}, Decision score = {score:.3f}")
