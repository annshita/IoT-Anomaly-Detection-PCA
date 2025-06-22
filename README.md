# IoT Anomaly Detection using PCA and Multiple Models

This project implements an effective anomaly detection framework for Internet of Things (IoT) environments, leveraging both feature reduction techniques and multiple supervised & unsupervised machine learning algorithms. It addresses the challenges of high-dimensional data and resource-constrained IoT devices.

---

## 📊 Datasets

Three datasets were used, each with varying dimensionality and characteristics:

| Dataset   | Initial Shape | Features After PCA      |
| --------- | ------------- | ----------------------- |
| Dataset 1 | (48,003, 25)  | 15 principal components |
| Dataset 2 | (125,973, 42) | 20 principal components |
| Dataset 3 | (157,800, 63) | 27 principal components |

**Preprocessing applied to all datasets:**

* Label encoding of categorical features
* Feature scaling (normalization)
* Dimensionality reduction using Principal Component Analysis (PCA) to retain 90% variance

---

## 🧪 Models Used

### 🧬 Unsupervised Models

#### 1. One-Class SVM (OCSVM)

* Defines a boundary around normal data to detect anomalies.
* Suitable for high-dimensional spaces.

#### 2. Isolation Forest (ISO-FOR)

* Randomly isolates anomalies by selecting random features and split values.
* Very efficient on large, high-dimensional datasets.

#### 3. Local Outlier Factor (LOF)

* Detects anomalies by comparing local density of points.
* Effective for complex, non-convex data distributions.

### 🔧 Supervised Models

#### 4. Decision Tree (DT)

* Interpretable, handles both categorical and numerical data.
* Performs well on high-dimensional structured data.

#### 5. K-Nearest Neighbors (KNN)

* Classifies based on proximity in feature space.
* Sensitive to feature scaling and may be computationally intensive on large datasets.

---

## 📈 Evaluation Metrics

All models were evaluated using labeled data based on:

* **Accuracy**: (TP + TN) / Total
* **Precision**: TP / (TP + FP)
* **Recall**: TP / (TP + FN)
* **F1 Score**: Harmonic mean of precision and recall
* **MSE**: Mean Squared Error
* **ROC-AUC**: Area Under Receiver Operating Characteristic Curve

---

## 📆 Key Findings

* **Isolation Forest** consistently outperformed other unsupervised methods, especially in high-dimensional settings, making it ideal for IoT data.
* **Local Outlier Factor (LOF)** handled density-based and non-convex anomalies very well.
* **One-Class SVM (OCSVM)** performed adequately but was more resource-intensive.
* **Decision Tree** was the most effective supervised model with excellent accuracy, precision, and recall.
* **KNN** performed well with clean separable data but struggled with high-dimensional and large-scale IoT data due to sensitivity to feature scaling.

---

## 💾 Technologies Used

* Python 3.x
* Scikit-learn
* NumPy
* Pandas
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 🔧 How to Run

1. Clone the repository:

```bash
git clone <your-repo-url>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare and preprocess datasets according to provided preprocessing scripts.

4. Run training and evaluation scripts:

```bash
python pca_preprocessing.py
python train_unsupervised_models.py
python train_supervised_models.py
```
---

This project demonstrates an effective approach to IoT anomaly detection using dimensionality reduction and a variety of machine learning techniques suitable for high-dimensional, real-world IoT data.
