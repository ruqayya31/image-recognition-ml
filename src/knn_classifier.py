"""
K-Nearest Neighbors (KNN) Classifier Implementation
Authors: MEHEK A, NASREEN T S, NAVJOT KAUR, SAYADA RUQAYYA
Institution: R.L. Jalappa Institute of Technology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                           recall_score, f1_score, classification_report)
from sklearn.neighbors import KNeighborsClassifier

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        tuple: Scaled features and target variable
    """
    # Load dataset
    hazel_df = pd.read_csv(file_path)
    print("Dataset shape:", hazel_df.shape)
    print("\nDataset head:")
    print(hazel_df.head())
    
    # Feature selection
    all_features = hazel_df.drop("CLASS", axis=1)
    target_feature = hazel_df["CLASS"]
    
    # Dataset preprocessing - MinMax Scaling
    x = all_features.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_features = pd.DataFrame(x_scaled, columns=all_features.columns)
    
    print("\nFeatures after scaling:")
    print(scaled_features.head())
    
    return scaled_features, target_feature

def find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 31)):
    """
    Find optimal k value for KNN classifier
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        k_range: Range of k values to test
    
    Returns:
        int: Optimal k value
    """
    accuracies = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o', linewidth=2, markersize=6)
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('K-NN: Finding Optimal K Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/knn_k_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    optimal_k = k_range[np.argmax(accuracies)]
    print(f"Optimal K value: {optimal_k} with accuracy: {max(accuracies):.4f}")
    
    return optimal_k

def train_knn_classifier(X_train, y_train, X_test, y_test, k=25):
    """
    Train KNN Classifier and make predictions
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        k: Number of neighbors
    
    Returns:
        tuple: Trained model and predictions
    """
    # Initialize KNN model
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    knn_pred = model.predict(X_test)
    
    return model, knn_pred

def evaluate_model(model, scaled_features, target_feature, y_test, knn_pred):
    """
    Evaluate the model performance using various metrics
    
    Args:
        model: Trained model
        scaled_features: All scaled features
        target_feature: Target variable
        y_test: True test labels
        knn_pred: Predicted test labels
    """
    # Cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    result_knn = cross_val_score(model, scaled_features, target_feature, 
                               cv=kfold, scoring='accuracy')
    
    print('=' * 60)
    print('K-NEAREST NEIGHBORS CLASSIFIER RESULTS')
    print('=' * 60)
    print(f'Overall Cross-Validation Score: {round(result_knn.mean()*100, 2)}%')
    print(f'Standard Deviation: {round(result_knn.std()*100, 2)}%')
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt=".0f", cmap='Greens')
    plt.title('K-NN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Fold accuracy visualization
    _result_knn = [r*100 for r in result_knn]
    plt.figure(figsize=(10, 6))
    plt.plot(_result_knn, marker='s', linewidth=2, markersize=6, color='green')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy (%)')
    plt.title('K-NN - K-Fold Cross Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim([min(_result_knn)-2, max(_result_knn)+2])
    plt.tight_layout()
    plt.savefig('results/knn_fold_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed metrics
    print('\n' + '='*50)
    print('DETAILED PERFORMANCE METRICS')
    print('='*50)
    
    print(f'Micro Precision: {precision_score(y_test, knn_pred, average="micro"):.4f}')
    print(f'Micro Recall: {recall_score(y_test, knn_pred, average="micro"):.4f}')
    print(f'Micro F1-score: {f1_score(y_test, knn_pred, average="micro"):.4f}\n')
    
    print(f'Macro Precision: {precision_score(y_test, knn_pred, average="macro"):.4f}')
    print(f'Macro Recall: {recall_score(y_test, knn_pred, average="macro"):.4f}')
    print(f'Macro F1-score: {f1_score(y_test, knn_pred, average="macro"):.4f}\n')
    
    print(f'Weighted Precision: {precision_score(y_test, knn_pred, average="weighted"):.4f}')
    print(f'Weighted Recall: {recall_score(y_test, knn_pred, average="weighted"):.4f}')
    print(f'Weighted F1-score: {f1_score(y_test, knn_pred, average="weighted"):.4f}')
    
    # Classification Report
    print('\n' + '='*60)
    print('K-NEAREST NEIGHBORS CLASSIFICATION REPORT')
    print('='*60)
    print(classification_report(y_test, knn_pred))

def main():
    """
    Main function to execute the KNN classification pipeline
    """
    # Set file path - Update this with your actual dataset path
    file_path = "data/dataset.csv"  # Update this path
    
    try:
        # Load and preprocess data
        scaled_features, target_feature = load_and_preprocess_data(file_path)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, target_feature, 
            test_size=0.25, 
            random_state=40, 
            stratify=target_feature
        )
        
        print(f"\nData split completed:")
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        
        # Find optimal k (optional - comment out if you want to use k=25 directly)
        print("\nFinding optimal K value...")
        optimal_k = find_optimal_k(X_train, y_train, X_test, y_test)
        
        # Train model with optimal k (or use default k=25)
        k_value = 25  # You can use optimal_k here if you ran the optimization
        print(f"\nTraining KNN with k={k_value}")
        model, knn_pred = train_knn_classifier(X_train, y_train, X_test, y_test, k=k_value)
        
        # Evaluate model
        evaluate_model(model, scaled_features, target_feature, y_test, knn_pred)
        
    except FileNotFoundError:
        print(f"Error: Could not find the dataset file at {file_path}")
        print("Please update the file_path variable with the correct path to your dataset.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if _name_ == "_main_":
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    main()
