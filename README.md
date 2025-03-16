# Codveda Machine Learning Submission

## 📌 Project Overview

This repository contains machine learning tasks as part of the [Codveda Technologies](https://www.linkedin.com/company/codveda-technologies/posts/?feedView=all) internship program. The tasks are divided into three levels: `Basic`, `Intermediate`, and `Advanced`. Each level includes specific machine learning concepts and techniques, implemented using Python and various ML libraries.

## 📂 Repository Structure

      ```bash
      codveda-ml-submission/
      │
      ├── Level 1 Basic/
      │   ├── Task_1_Data_Preprocessing/
      │   │   ├── data/
      │   │   │   ├── processed/
      │   │   │   └── raw/
      │   │   └── preprocessing.ipynb
      │   │
      │   ├── Task_2_Linear_Regression/
      │   │   ├── data/
      │   │   └── linear_regression.ipynb
      │   │
      │   └── Task_3_KNN_Classifier/
      │       ├── data/
      │       └── knn_classifier.ipynb
      │
      ├── Level 2 Intermediate/
      │   ├── Task_1_Logistic_Regression/
      │   │   ├── data/
      │   │   └── logistic_regression.ipynb
      │   │
      │   ├── Task_2_Decision_Tree/
      │   │   ├── data/
      │   │   └── decision_tree.ipynb
      │   │
      │   └── Task_3_KMeans_Clustering/
      │       ├── data/
      │       └── kmeans_clustering.ipynb
      │
      ├── Level 3 Advanced/
      │   ├── Task_1_Random_Forest/
      │   │   ├── data/
      │   │   └── random_forest.ipynb
      │   │
      │   ├── Task_2_SVM/
      │   │   ├── data/
      │   │   └── svm.ipynb
      │   │
      │   └── Task_3_Neural_Network/
      │       ├── data/
      │       └── neural_network.ipynb
      │
      ├── .gitignore
      │
      ├── LICENSE
      │
      └── README.md
      ```

## 🎯 Task Breakdown

### Level 1 (Basic)

#### Task 1: Data Preprocessing for Machine Learning

- Handle missing data (mean/median imputation, dropping).
- Encode categorical variables (one-hot encoding, label encoding).
- Normalize or standardize numerical features.
- Split dataset into training and testing sets.
- Tools: `Python, Pandas, Scikit-learn`

#### Task 2: Build a Simple Linear Regression Model

- Load and preprocess dataset.
- Train a linear regression model using `Scikit-learn`.
- Interpret model coefficients.
- Evaluate model with R-squared and Mean Squared Error (MSE).
- Tools: `Python, Pandas, Scikit-Learn`

#### Task 3: Implement K-Nearest Neighbors (KNN) Classifier

- Train a KNN model on labeled dataset.
- Evaluate using accuracy, confusion matrix, precision, recall.
- Compare results using different values of K.
- Tools: `Python, Scikit-Learn, Pandas`

### Level 2 (Intermediate)

#### Task 1: Logistic Regression for Binary Classification

- Load and preprocess dataset.
- Train logistic regression model.
- Interpret model coefficients and odds ratio.
- Evaluate using accuracy, precision, recall, and ROC curve.
- Tools: `Python, Pandas, Scikit-Learn, Matplotlib`

#### Task 2: Decision Trees for Classification

- Train decision tree classifier on labeled dataset.
- Visualize the tree structure.
- Prune tree to prevent overfitting.
- Evaluate using classification metrics (accuracy, F1-score).
- Tools: `Python, Scikit-Learn, Pandas, Matplotlib`

#### Task 3: K-Means Clustering

- Load and preprocess dataset (scaling).
- Apply K-Means clustering and determine optimal cluster number (elbow method).
- Visualize clusters using scatter plots.
- Interpret clustering results.
- Tools: `Python, Scikit-learn, Matplotlib, Seaborn`

### Level 3 (Advanced)

#### Task 1: Build a Random Forest Classifier

- Train Random Forest model and tune hyperparameters.
- Evaluate using cross-validation, precision, recall, F1-score.
- Perform feature importance analysis.
- Tools: `Python, Scikit-learn, Pandas, Matplotlib`

#### Task 2: Support Vector Machine (SVM) for Classification

- Train SVM model on labeled dataset.
- Use different kernels (linear, RBF) and compare performance.
- Visualize decision boundary.
- Evaluate using accuracy, precision, recall, and AUC.
- Tools: `Python, Scikit-learn, Pandas, Matplotlib`

#### Task 3: Neural Network for Classification

- Load and preprocess dataset.
- Design neural network (input, hidden, output layers).
- Train model using backpropagation.
- Evaluate using accuracy and visualize training/validation loss.
- Tools: `Python, TensorFlow/Keras, Pandas, Matplotlib`

## 🛠 Dependencies

Ensure you have the following installed before running the notebooks:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras
```

## 🚀 How to Run the Code

1. **Clone the repository**:

   ```bash
   git clone https://github.com/RyanGA09/codveda-ml-submission.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd codveda-ml-submission
   ```

3. **Open Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

4. **Run the notebooks in the `Level_1_Basic`, `Level_2_Intermediate`, and `Level_3_Advanced` folders.**

## 📧 Contact

For any queries or collaborations, reach out via:

- **LinkedIn:** [Ryan Gading Abdullah](https://linkedin.com/in/ryan-gading-abdullah)
- **GitHub:** [RyanGA09](https://github.com/RyanGA09)
- **Instagram:** [Ryan Gading](https://www.instagram.com/ryan_g._a/)

## LICENSE

[MIT LICENSE](LICENSE)

&copy;2025 Ryan Gading Abdullah. All rights reserved.
