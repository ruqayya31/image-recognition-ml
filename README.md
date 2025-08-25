# Image Recognition with Machine Learning Algorithm

## 📋 Project Overview
This project implements and compares three traditional machine learning algorithms for image recognition and classification. We analyze the performance of Decision Tree, K-Nearest Neighbors (KNN), and Naive Bayes classifiers on a tree nuts dataset to determine the most effective approach for image classification tasks.

## 🎯 Objectives
- Implement multiple machine learning algorithms for image classification
- Compare performance across Decision Tree, KNN, and Naive Bayes algorithms
- Analyze model effectiveness using comprehensive evaluation metrics
- Generate visualizations and performance comparisons
- Provide insights into which algorithm works best for this classification task

## 📊 Dataset Setup

**Important:** The dataset is not included in this repository due to file size limitations.

### Download Instructions:
1. Go to [Tree Nuts Classification Dataset on Kaggle](https://www.kaggle.com/datasets/gpiosenka/tree-nuts-image-classification)
2. Sign in to Kaggle (create free account if needed)
3. Click "Download" button
4. Extract the ZIP file and find the main CSV file
5. Rename it to `tree_nuts_dataset.csv`
6. Place it in the `data/` folder
7. See `data/README.md` for detailed step-by-step instructions

### Alternative Dataset:
You can also use the [Pistachio Dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset) if preferred.

### Dataset Features:
- Multiple image-derived features for classification
- Target column: "CLASS" for different categories
- CSV format ready for traditional ML algorithms
- Suitable for multi-class classification tasks

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and metrics
- **matplotlib** - Data visualization and plotting
- **seaborn** - Statistical data visualization and heatmaps

## 📁 Project Structure
```
image-recognition-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                    # Dataset download instructions
│   └── tree_nuts_dataset.csv       # Dataset (download required)
├── decision_tree_classifier.py     # Decision Tree implementation
├── knn_classifier.py               # K-Nearest Neighbors implementation
├── naive_bayes_classifier.py       # Naive Bayes implementation
└── results/                         # Generated results and visualizations
    ├── confusion_matrices/
    ├── accuracy_plots/
    └── performance_reports/
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ruqayya31/image-recognition-ml.git
   cd image-recognition-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and setup dataset:**
   - Follow the dataset download instructions above
   - Place the CSV file in the `data/` folder
   - Verify the file is named `tree_nuts_dataset.csv`

## 🏃‍♂️ How to Run

### Run Individual Classifiers:
```bash
# Run Decision Tree classifier
python decision_tree_classifier.py

# Run K-Nearest Neighbors classifier  
python knn_classifier.py

# Run Naive Bayes classifier
python naive_bayes_classifier.py
```

### What Each Script Does:
- Loads and preprocesses the dataset
- Trains the respective classifier
- Performs 10-fold cross-validation
- Generates confusion matrices and accuracy plots
- Calculates comprehensive performance metrics
- Saves all results to the `results/` folder

## 📈 Machine Learning Models

### 1. Decision Tree Classifier
- **Algorithm:** CART (Classification and Regression Trees)
- **Parameters:** 
  - Criterion: Gini impurity
  - Min samples split: 10
  - Min samples leaf: 1
  - Max features: Auto
- **Strengths:** Highly interpretable, handles non-linear relationships
- **Use Case:** When you need to understand feature importance

### 2. K-Nearest Neighbors (KNN)
- **Algorithm:** Distance-based classification
- **Parameters:** 
  - Number of neighbors: 25
  - Distance metric: Euclidean
- **Strengths:** Simple, non-parametric, effective with sufficient data
- **Use Case:** When local patterns in data are important

### 3. Naive Bayes Classifier
- **Algorithm:** Gaussian Naive Bayes
- **Assumption:** Features are conditionally independent given the class
- **Strengths:** Fast training and prediction, works well with small datasets
- **Use Case:** When you need quick baseline results

## 🔍 Evaluation Metrics

Each model is comprehensively evaluated using:
- **10-fold Cross-validation Accuracy**
- **Confusion Matrices** with heatmap visualizations
- **Precision, Recall, F1-score** (Micro, Macro, Weighted averages)
- **Classification Reports** with per-class metrics
- **Accuracy Visualization** across all folds

## 📊 Results

After running the classifiers, you'll find:
- **Confusion matrices** saved as high-resolution PNG files
- **Cross-validation accuracy plots** showing performance across folds
- **Performance metrics** saved in detailed text reports
- **All visualizations** automatically saved in the `results/` folder

### Performance Summary:
| Classifier | Accuracy | Status |
|------------|----------|---------|
| Decision Tree | Run to see results | ⏳ Pending |
| K-Nearest Neighbors | Run to see results | ⏳ Pending |
| Naive Bayes | Run to see results | ⏳ Pending |

*Update this table with actual results after running the models*

## 📈 Expected Outcomes
- Performance comparison between three different ML approaches
- Insights into which algorithm works best for tree nuts classification
- Visual analysis through confusion matrices and accuracy plots
- Comprehensive metric analysis for informed model selection

## 🔮 Future Enhancements
- Implement deep learning approaches (Convolutional Neural Networks)
- Add hyperparameter tuning using GridSearchCV
- Experiment with ensemble methods (Random Forest, Gradient Boosting)
- Include feature selection and engineering techniques
- Add real-time prediction capabilities with a web interface

## 👥 Team Members
- **Mehek A** [1RL22CD028]
- **Nasreen T S** [1RL22CD032]  
- **Navjot Kaur** [1RL22CD033]
- **Sayada Ruqayya** [1RL22CD047]

**Project Guide:** Dr. Mrutyunjaya M S, Associate Professor

## 🏫 Institution
**R. L. Jalappa Institute of Technology**  
Department of Computer Science Engineering (Data Science)  
Doddaballapur, Bangalore Rural District - 561 203  
Academic Year: 2023-24

## 🤝 Contributing
This project is created for educational purposes as part of the 5th Semester Mini-Project requirement. Feel free to fork and extend for learning purposes.

## 📄 License
This project is for academic and educational purposes only.

## 🙏 Acknowledgments
- Kaggle for providing the dataset
- Scikit-learn community for excellent machine learning tools
- Our guide Dr. Mrutyunjaya M S for valuable guidance
- R. L. Jalappa Institute of Technology for project support

## 📞 Contact
For any questions regarding this project, please reach out through GitHub issues or contact the team members through the institution.

---
*This project demonstrates the implementation and comparison of traditional machine learning algorithms for image classification tasks, providing insights into model performance and selection criteria.*
