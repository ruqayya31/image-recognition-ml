"""
Naive Bayes Classifier Implementation
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
from sklearn.naive_bayes import GaussianNB

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

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """
    Train Gaussian Naive Bayes Classifier and make predictions
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
    
    Returns:
        tuple: Trained model and predictions
    """
    # Initialize Gaussian Naive Bayes model
    model = GaussianNB()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    nb_pred = model.predict(X_test)
    
    return model, nb_pred

def plot_feature_distributions(X_train, y_train):
    """
    Plot feature distributions for different classes
    
    Args:
        X_train: Training features
        y_train: Training labels
    """
    # Convert to DataFrame for easier plotting
    train_data = pd.DataFrame(X_train)
    train_data['CLASS'] = y_train
    
    # Plot distributions for first few features
    n_features_to_plot = min(6, len(train_data.columns)-1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(n_features_to_plot):
        for class_label in train_data['CLASS'].unique():
            class_data = train_data[train_data['CLASS'] == class_label][i]
            axes[i].hist(class_data, alpha=0.6, label=f'Class {class_label}', bins=20)
        
        axes[i].set_title(f'Feature {i} Distribution')
        axes[i].set_xlabel('Feature Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/naive_bayes_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, scaled_features, target_feature, y_test, nb_pred):
    """
    Evaluate the model performance using various metrics
    
    Args:
        model: Trained model
        scaled_features: All scaled features
        target_feature: Target variable
        y_test: True test labels
        nb_pred: Predicted test labels
    """
    # Cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    result_nb = cross_val_score(model, scaled_features, target_feature, 
                              cv=kfold, scoring='accuracy')
    
    print('=' * 60)
    print('GAUSSIAN NAIVE BAYES CLASSIFIER RESULTS')
    print('=' * 60)
    print(f'Overall Cross-Validation Score: {round(result_nb.mean()*100, 2)}%')
    print(f'Standard Deviation: {round(result_nb.std()*100, 2)}%')
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, nb_pred), annot=True, fmt=".0f", cmap='Oranges')
    plt.title('Naive Bayes Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/naive_bayes_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Fold accuracy visualization
    _result_nb = [r*100 for r in result_nb]
    plt.figure(figsize=(10, 6))
    plt.plot(_result_nb, marker='^', linewidth=2, markersize=6, color='orange')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy (%)')
    plt.title('Naive Bayes - K-Fold Cross Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim([min(_result_nb)-2, max(_result_nb)+2])
    plt.tight_layout()
    plt.savefig('results/naive_bayes_fold_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed metrics
    print('\n' + '='*50)
    print('DETAILED PERFORMANCE METRICS')
    print('='*50)
    
    print(f'Micro Precision: {precision_score(y_test, nb_pred, average="micro"):.4f}')
    print(f'Micro Recall: {recall_score(y_test, nb_pred, average="micro"):.4f}')
    print(f'Micro F1-score: {f1_score(y_test, nb_pred, average="micro"):.4f}\n')
    
    print(f'Macro Precision: {precision_score(y_test, nb_pred, average="macro"):.4f}')
    print(f'Macro Recall: {recall_score(y_test, nb_pred, average="macro"):.4f}')
    print(f'Macro F1-score: {f1_score(y_test, nb_pred, average="macro"):.4f}\n')
    
    print(f'Weighted Precision: {precision_score(y_test, nb_pred, average="weighted"):.4f}')
    print(f'Weighted Recall: {recall_score(y_test, nb_pred, average="weighted"):.4f}')
    print(f'Weighted F1-score: {f1_score(y_test, nb_pred, average="weighted"):.4f}')
    
    # Classification Report
    print('\n' + '='*60)
    print('NAIVE BAYES CLASSIFICATION REPORT')
    print('='*60)
    print(classification_report(y_test, nb_pred))

def main():
    """
    Main function to execute the Naive Bayes classification pipeline
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
        
        # Plot feature distributions
        print("\nPlotting feature distributions...")
        plot_feature_distributions(X_train, y_train)
        
        # Train model
        model, nb_pred = train_naive_bayes(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        evaluate_model(model, scaled_features, target_feature, y_test, nb_pred)
        
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
