# Image Recognition with Machine Learning Algorithm

## 📋 Project Overview

This mini-project implements various machine learning algorithms for image recognition, focusing on comparing the performance of different classifiers including Decision Tree, K-Nearest Neighbors (KNN), and Naive Bayes.

## 🎓 Academic Information

- *University*: Visvesvaraya Technological University, Belagavi
- *Department*: CSE (Data Science)
- *Semester*: 7th Semester (2025-26)
- *Institution*: R.L. Jalappa Institute of Technology, Doddaballapur

## 👥 Team Members

- *MEHEK A* [1RL22CD028]
- *NASREEN T S* [1RL22CD032] 
- *NAVJOT KAUR* [1RL22CD033]
- *SAYADA RUQAYYA* [1RL22CD047]

*Guide*: Dr. Mrutyunjaya M S, Associate Professor

## 🔬 Abstract

Image recognition using deep learning has become a cornerstone of modern computer vision, enabling machines to achieve human-level accuracy in identifying and classifying objects within images. This project explores various machine learning techniques including:

- *Convolutional Neural Networks (CNNs)* - For spatial feature extraction
- *Recurrent Neural Networks (RNNs)* - For sequential data processing
- *Generative Adversarial Networks (GANs)* - For generative tasks
- *Hybrid Models* - Combining multiple architectures

## 🛠 Technologies Used

- *Python* - Primary programming language
- *Scikit-learn* - Machine learning library
- *Pandas* - Data manipulation and analysis
- *NumPy* - Numerical computing
- *Matplotlib* - Data visualization
- *Seaborn* - Statistical data visualization

## 📊 Implemented Algorithms

### 1. Decision Tree Classifier
- Uses Gini criterion for splitting
- Configurable parameters for min_samples_split and min_samples_leaf
- Includes cross-validation with K-fold (k=10)

### 2. K-Nearest Neighbors (KNN)
- Implemented with n_neighbors = 25
- Distance-based classification
- Performance evaluation with confusion matrix

### 3. Naive Bayes Classifier
- Gaussian Naive Bayes implementation
- Probabilistic classification approach
- Cross-validation scoring

## 🏗 Project Structure


image-recognition-ml/
├── src/
│   ├── decision_tree.py
│   ├── knn_classifier.py
│   ├── naive_bayes.py
│   └── data_preprocessing.py
├── data/
│   └── dataset.csv
├── results/
│   ├── confusion_matrices/
│   └── performance_metrics/
├── docs/
│   └── project_report.pdf
├── requirements.txt
├── README.md
└── .gitignore


## 🚀 Getting Started

### Prerequisites

bash
pip install -r requirements.txt


### Installation

1. Clone the repository
bash
git clone https://github.com/yourusername/image-recognition-ml.git
cd image-recognition-ml


2. Install dependencies
bash
pip install -r requirements.txt


3. Run the algorithms
bash
python src/decision_tree.py
python src/knn_classifier.py
python src/naive_bayes.py


## 📈 Results and Performance

The project evaluates each classifier using:
- *Accuracy Score* - Overall classification accuracy
- *Confusion Matrix* - Classification performance visualization
- *Precision, Recall, F1-Score* - Detailed performance metrics
- *Cross-validation* - Model stability assessment

### Performance Metrics Evaluated:
- Micro Precision/Recall/F1-score
- Macro Precision/Recall/F1-score  
- Weighted Precision/Recall/F1-score

## 🔍 Methodology

1. *Data Preprocessing*
   - Feature selection and scaling using MinMaxScaler
   - Train-test split (75:25 ratio)

2. *Model Training*
   - 10-fold cross-validation
   - Parameter optimization

3. *Evaluation*
   - Performance metrics calculation
   - Visualization of results

## 🎯 Future Scope

- *Efficiency Improvement*: Enhance computational efficiency of RNNs and GANs
- *Training Stability*: Implement advanced strategies for GAN training stability
- *Hybrid Model Optimization*: Reduce complexity while maintaining functionality
- *Transfer Learning*: Apply knowledge across domains
- *Real-time Processing*: Enable efficient real-time image recognition

## 📚 References

1. Krizhevsky, A., et al. (2012). ImageNet classification with deep convolutional neural networks.
2. Donahue, J., et al. (2015). Long-term recurrent convolutional networks for visual recognition.
3. Goodfellow, I., et al. (2014). Generative adversarial nets.
4. Franco-Duarte R., et al. (2019). Advances in chemical and biological methods to identify microorganisms.

## 📄 License

This project is part of academic coursework at VTU and is intended for educational purposes.

## 🤝 Contributing

This is an academic project. For any queries or suggestions, please contact the team members.

## 📧 Contact

For questions about this project, please reach out to the team members.

*Note*: This project was developed as part of the V Semester curriculum for Bachelor of Engineering in CSE (Data Science) at R.L. Jalappa Institute of Technology.
