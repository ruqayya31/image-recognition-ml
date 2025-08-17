"""
Model Comparison Script
Authors: MEHEK A, NASREEN T S, NAVJOT KAUR, SAYADA RUQAYYA
Institution: R.L. Jalappa Institute of Technology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """
    Compare multiple machine learning models for image recognition
    """
    
    def _init_(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self, file_path, target_column='CLASS'):
        """
        Load and preprocess the dataset
        """
        # Load dataset
        data = pd.read_csv(file_path)
        print(f"Dataset loaded: {data.shape}")
        
        # Separate features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Scale features
        scaler = preprocessing.MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        
    def initialize_models(self):
        """
        Initialize all models to be compared
        """
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                criterion='gini',
                min_samples_split=10,
                min_samples_leaf=1,
                max_features='auto',
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=25,
                weights='uniform',
                algorithm='auto'
            ),
            'Naive Bayes': GaussianNB()
        }
        
        print("Models initialized:")
        for name in self.models.keys():
            print(f"- {name}")
    
    def train_and_evaluate_models(self):
        """
        Train and evaluate all models
        """
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING MODELS")
        print("="*60)
        
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=kfold, scoring='accuracy')
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision_micro = precision_score(self.y_test, y_pred, average='micro')
            precision_macro = precision_score(self.y_test, y_pred, average='macro')
            precision_weighted = precision_score(self.y_test, y_pred, average='weighted')
            
            recall_micro = recall_score(self.y_test, y_pred, average='micro')
            recall_macro = recall_score(self.y_test, y_pred, average='macro')
            recall_weighted = recall_score(self.y_test, y_pred, average='weighted')
            
            f1_micro = f1_score(self.y_test, y_pred, average='micro')
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision_micro': precision_micro,
                'precision_macro': precision_macro,
                'precision_weighted': precision_weighted,
                'recall_micro': recall_micro,
                'recall_macro': recall_macro,
                'recall_weighted': recall_weighted,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'cv_scores': cv_scores
            }
            
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"Test accuracy: {accuracy:.4f}")
    
    def create_comparison_plots(self):
        """
        Create comprehensive comparison plots
        """
        print("\nCreating comparison visualizations...")
        
        # 1. Accuracy Comparison Bar Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # CV Accuracy comparison
        models_names = list(self.results.keys())
        cv_means = [self.results[name]['cv_mean'] for name in models_names]
        cv_stds = [self.results[name]['cv_std'] for name in models_names]
        
        axes[0,0].bar(models_names, cv_means, yerr=cv_stds, capsize=5, 
                     color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0,0].set_title('Cross-Validation Accuracy Comparison')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim([0, 1])
        for i, v in enumerate(cv_means):
            axes[0,0].text(i, v + cv_stds[i] + 0.01, f'{v:.3f}', ha='center')
        
        # Test Accuracy comparison
        test_accuracies = [self.results[name]['accuracy'] for name in models_names]
        axes[0,1].bar(models_names, test_accuracies, 
                     color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0,1].set_title('Test Set Accuracy Comparison')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_ylim([0, 1])
        for i, v in enumerate(test_accuracies):
            axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # F1-Score comparison
        f1_scores = [self.results[name]['f1_weighted'] for name in models_names]
        axes[1,0].bar(models_names, f1_scores, 
                     color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1,0].set_title('F1-Score (Weighted) Comparison')
        axes[1,0].set_ylabel('F1-Score')
        axes[1,0].set_ylim([0, 1])
        for i, v in enumerate(f1_scores):
            axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Precision-Recall comparison
        precisions = [self.results[name]['precision_weighted'] for name in models_names]
        recalls = [self.results[name]['recall_weighted'] for name in models_names]
        
        x = np.arange(len(models_names))
        width = 0.35
        
        axes[1,1].bar(x - width/2, precisions, width, label='Precision', 
                     color='lightblue')
        axes[1,1].bar(x + width/2, recalls, width, label='Recall', 
                     color='lightgreen')
        axes[1,1].set_title('Precision vs Recall Comparison')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(models_names)
        axes[1,1].legend()
        axes[1,1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Cross-validation scores distribution
        plt.figure(figsize=(12, 6))
        cv_data = []
        labels = []
        for name in models_names:
            cv_data.extend(self.results[name]['cv_scores'])
            labels.extend([name] * len(self.results[name]['cv_scores']))
        
        cv_df = pd.DataFrame({'Model': labels, 'CV_Score': cv_data})
        sns.boxplot(data=cv_df, x='Model', y='CV_Score')
        plt.title('Cross-Validation Score Distribution')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/cv_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, name in enumerate(models_names):
            cm = confusion_matrix(self.y_test, self.results[name]['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_comparison_table(self):
        """
        Create detailed comparison table of all metrics
        """
        print("\n" + "="*80)
        print("DETAILED MODEL COMPARISON TABLE")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for name, results in self.results.items():
            row = {
                'Model': name,
                'CV_Mean': f"{results['cv_mean']:.4f}",
                'CV_Std': f"{results['cv_std']:.4f}",
                'Test_Accuracy': f"{results['accuracy']:.4f}",
                'Precision_Micro': f"{results['precision_micro']:.4f}",
                'Precision_Macro': f"{results['precision_macro']:.4f}",
                'Precision_Weighted': f"{results['precision_weighted']:.4f}",
                'Recall_Micro': f"{results['recall_micro']:.4f}",
                'Recall_Macro': f"{results['recall_macro']:.4f}",
                'Recall_Weighted': f"{results['recall_weighted']:.4f}",
                'F1_Micro': f"{results['f1_micro']:.4f}",
                'F1_Macro': f"{results['f1_macro']:.4f}",
                'F1_Weighted': f"{results['f1_weighted']:.4f}"
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('results/model_comparison_table.csv', index=False)
        print(f"\nDetailed comparison table saved to: results/model_comparison_table.csv")
        
        # Find best model for each metric
        print("\n" + "="*60)
        print("BEST PERFORMING MODELS BY METRIC")
        print("="*60)
        
        metrics = ['cv_mean', 'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        metric_names = ['Cross-Validation', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric, metric_name in zip(metrics, metric_names):
            best_model = max(self.results.items(), key=lambda x: x[1][metric])
            print(f"{metric_name}: {best_model[0]} ({best_model[1][metric]:.4f})")
    
    def generate_classification_reports(self):
        """
        Generate detailed classification reports for each model
        """
        print("\n" + "="*80)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*80)
        
        for name, results in self.results.items():
            print(f"\n{name.upper()} CLASSIFICATION REPORT")
            print("-" * (len(name) + 25))
            print(classification_report(self.y_test, results['predictions']))
    
    def run_comparison_pipeline(self, file_path, target_column='CLASS'):
        """
        Run complete model comparison pipeline
        """
        print("="*80)
        print("MODEL COMPARISON PIPELINE")
        print("="*80)
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data(file_path, target_column)
            
            # Step 2: Initialize models
            self.initialize_models()
            
            # Step 3: Train and evaluate models
            self.train_and_evaluate_models()
            
            # Step 4: Create visualizations
            self.create_comparison_plots()
            
            # Step 5: Create detailed comparison table
            self.create_detailed_comparison_table()
            
            # Step 6: Generate classification reports
            self.generate_classification_reports()
            
            print("\n" + "="*80)
            print("MODEL COMPARISON PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            # Recommend best model
            best_overall = max(self.results.items(), 
                             key=lambda x: (x[1]['cv_mean'] + x[1]['accuracy']) / 2)
            
            print(f"\nRECOMMENDED MODEL: {best_overall[0]}")
            print(f"Average Score: {(best_overall[1]['cv_mean'] + best_overall[1]['accuracy']) / 2:.4f}")
            
        except Exception as e:
            print(f"Error in comparison pipeline: {str(e)}")

def main():
    """
    Main function to run model comparison
    """
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Set file path
    file_path = "data/dataset.csv"  # Update with your dataset path
    
    # Run comparison pipeline
    comparator.run_comparison_pipeline(file_path)

if _name_ == "_main_":
    main()
