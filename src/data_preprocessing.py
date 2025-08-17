"""
Data Preprocessing Utilities
Authors: MEHEK A, NASREEN T S, NAVJOT KAUR, SAYADA RUQAYYA
Institution: R.L. Jalappa Institute of Technology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for machine learning workflows
    """
    
    def _init_(self):
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, file_path, target_column='CLASS'):
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to CSV file
            target_column (str): Name of target column
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        try:
            data = pd.read_csv(file_path)
            self.target_column = target_column
            print(f"Dataset loaded successfully!")
            print(f"Dataset shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def explore_data(self, data):
        """
        Perform exploratory data analysis
        
        Args:
            data (pandas.DataFrame): Dataset to explore
        """
        print("="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print(f"Dataset shape: {data.shape}")
        print(f"Number of features: {data.shape[1]-1}")
        print(f"Number of samples: {data.shape[0]}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(data.head())
        
        # Data types
        print("\nData types:")
        print(data.dtypes)
        
        # Missing values
        print("\nMissing values:")
        missing_values = data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(data.describe())
        
        # Target distribution
        if self.target_column in data.columns:
            print(f"\n{self.target_column} distribution:")
            print(data[self.target_column].value_counts())
            
            # Plot target distribution
            plt.figure(figsize=(10, 6))
            data[self.target_column].value_counts().plot(kind='bar')
            plt.title(f'{self.target_column} Distribution')
            plt.xlabel(self.target_column)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/target_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def handle_missing_values(self, data, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            data (pandas.DataFrame): Dataset with potential missing values
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
            
        Returns:
            pandas.DataFrame: Dataset with imputed values
        """
        # Separate features and target
        if self.target_column in data.columns:
            X = data.drop(self.target_column, axis=1)
            y = data[self.target_column]
        else:
            X = data
            y = None
        
        # Check for missing values
        if X.isnull().sum().sum() > 0:
            print(f"Handling missing values using {strategy} strategy...")
            
            # Initialize imputer
            if strategy in ['mean', 'median']:
                self.imputer = SimpleImputer(strategy=strategy)
            else:
                self.imputer = SimpleImputer(strategy='most_frequent')
            
            # Fit and transform features
            X_imputed = self.imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns)
            
            print("Missing values handled successfully!")
        else:
            print("No missing values found.")
        
        # Combine features and target
        if y is not None:
            data_cleaned = pd.concat([X, y], axis=1)
        else:
            data_cleaned = X
            
        return data_cleaned
    
    def remove_outliers(self, data, method='iqr', threshold=1.5):
        """
        Remove outliers from the dataset
        
        Args:
            data (pandas.DataFrame): Dataset
            method (str): Method to detect outliers ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pandas.DataFrame: Dataset without outliers
        """
        print(f"Removing outliers using {method} method...")
        
        # Separate features and target
        if self.target_column in data.columns:
            X = data.drop(self.target_column, axis=1)
            y = data[self.target_column]
        else:
            X = data
            y = None
        
        initial_shape = X.shape[0]
        
        if method == 'iqr':
            # IQR method
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outliers
            outliers = ((X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))).any(axis=1)
            
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(X))
            outliers = (z_scores > threshold).any(axis=1)
        
        # Remove outliers
        X_clean = X[~outliers]
        if y is not None:
            y_clean = y[~outliers]
            data_clean = pd.concat([X_clean, y_clean], axis=1)
        else:
            data_clean = X_clean
        
        final_shape = X_clean.shape[0]
        removed_count = initial_shape - final_shape
        
        print(f"Removed {removed_count} outliers ({removed_count/initial_shape*100:.2f}%)")
        print(f"Dataset shape after outlier removal: {data_clean.shape}")
        
        return data_clean
    
    def scale_features(self, X_train, X_test, method='minmax'):
        """
        Scale features using specified method
        
        Args:
            X_train (pandas.DataFrame): Training features
            X_test (pandas.DataFrame): Testing features
            method (str): Scaling method ('minmax', 'standard', 'robust')
            
        Returns:
            tuple: Scaled training and testing features
        """
        print(f"Scaling features using {method} scaler...")
        
        if method == 'minmax':
            self.scaler = preprocessing.MinMaxScaler()
        elif method == 'standard':
            self.scaler = preprocessing.StandardScaler()
        elif method == 'robust':
            self.scaler = preprocessing.RobustScaler()
        else:
            raise ValueError("Method must be 'minmax', 'standard', or 'robust'")
        
        # Fit scaler on training data and transform both sets
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print("Feature scaling completed!")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, data, test_size=0.25, random_state=42, stratify=True):
        """
        Split data into training and testing sets
        
        Args:
            data (pandas.DataFrame): Complete dataset
            test_size (float): Proportion of test set
            random_state (int): Random seed
            stratify (bool): Whether to stratify split based on target
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        # Perform train-test split
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        print(f"Training target distribution:")
        print(y_train.value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def create_feature_correlation_matrix(self, data):
        """
        Create and visualize feature correlation matrix
        
        Args:
            data (pandas.DataFrame): Dataset
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], 
                                          corr_matrix.columns[j], 
                                          corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print("\nHighly correlated feature pairs (|correlation| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"{feat1} - {feat2}: {corr:.3f}")
        else:
            print("\nNo highly correlated feature pairs found.")
    
    def preprocess_pipeline(self, file_path, target_column='CLASS', 
                          test_size=0.25, scaling_method='minmax',
                          handle_outliers=False, outlier_method='iqr'):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path (str): Path to dataset
            target_column (str): Name of target column
            test_size (float): Test set proportion
            scaling_method (str): Feature scaling method
            handle_outliers (bool): Whether to remove outliers
            outlier_method (str): Outlier detection method
            
        Returns:
            tuple: Preprocessed training and testing data
        """
        print("="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        data = self.load_data(file_path, target_column)
        if data is None:
            return None
        
        # Step 2: Explore data
        self.explore_data(data)
        
        # Step 3: Handle missing values
        data = self.handle_missing_values(data)
        
        # Step 4: Remove outliers (optional)
        if handle_outliers:
            data = self.remove_outliers(data, method=outlier_method)
        
        # Step 5: Create correlation matrix
        self.create_feature_correlation_matrix(data.drop(target_column, axis=1))
        
        # Step 6: Split data
        X_train, X_test, y_train, y_test = self.split_data(data, test_size=test_size)
        
        # Step 7: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, scaling_method)
        
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

# Example usage
if _name_ == "_main_":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    file_path = "data/dataset.csv"  # Update with your dataset path
    
    try:
        result = preprocessor.preprocess_pipeline(
            file_path=file_path,
            target_column='CLASS',
            test_size=0.25,
            scaling_method='minmax',
            handle_outliers=True,
            outlier_method='iqr'
        )
        
        if result is not None:
            X_train_scaled, X_test_scaled, y_train, y_test = result
            print(f"\nFinal preprocessed data shapes:")
            print(f"X_train: {X_train_scaled.shape}")
            print(f"X_test: {X_test_scaled.shape}")
            print(f"y_train: {y_train.shape}")
            print(f"y_test: {y_test.shape}")
            
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        print("Please update the file_path variable with the correct path to your dataset.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
