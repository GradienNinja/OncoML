# OncoML - Breast Cancer Diagnosis

Binary classification model using Support Vector Machine to predict malignant and benign tumors from diagnostic features.

## Results

**95.61% Accuracy | 95.35% Recall | 93.18% Precision**

| Metric | Score |
|--------|-------|
| Accuracy | 95.61% |
| Precision | 93.18% |
| Recall | 95.35% |
| F1-Score | 94.25% |
| AUC-ROC | 95.56% |

Confusion Matrix: `[[68, 3], [2, 41]]`
- 68 True Negatives (correct benign)
- 41 True Positives (correct malignant)  
- 3 False Positives (benign called malignant)
- 2 False Negatives (malignant called benign)

## Data

**UCI Breast Cancer Wisconsin Dataset**
- 569 samples (357 benign, 212 malignant)
- 30 diagnostic features
- Binary target: M (malignant) = 1, B (benign) = 0

Download: [Kaggle - UCI Breast Cancer](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/)

## Model

**Support Vector Machine with Linear Kernel**
```python
from sklearn.svm import SVC
svm = SVC(kernel="linear", C=1)
```

## Code Walkthrough

### 1. Load & Prepare Data
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
df = df.drop(columns=["id","Unnamed: 32"], errors='ignore')

x = df.drop("diagnosis", axis=1)
y = df['diagnosis'].map({"M": 1, "B": 0})
```

### 2. Train-Test Split
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)
```

### 3. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

### 4. Train SVM
```python
from sklearn.svm import SVC

svm = SVC(kernel="linear", C=1)
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
```

### 5. Evaluate
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
```

## Make Predictions

```python
# Define a sample with 30 features
sample = [
    18.5, 20.1, 121.0, 1050.0, 0.115, 0.26, 0.29, 0.14, 0.24, 0.078,
    1.1, 1.3, 7.5, 95.0, 0.006, 0.05, 0.06, 0.02, 0.02, 0.003,
    25.0, 26.0, 170.0, 1900.0, 0.15, 0.6, 0.7, 0.26, 0.45, 0.12
]

# Scale using same scaler
sample_scaled = scaler.transform(np.array(sample).reshape(1, -1))

# Predict
prediction = svm.predict(sample_scaled)[0]
diagnosis = "Malignant" if prediction == 1 else "Benign"
print(diagnosis)
```

## Key Metrics Explained

**Recall (95.35%)**: Catches 95 out of 100 actual cancers  
**Precision (93.18%)**: When predicting cancer, correct 93% of the time  
**Only 2 False Negatives**: Misses only 2 malignant cases (safe for clinical use)

## Usage

### Google Colab
1. Create new notebook: https://colab.research.google.com/
2. Upload dataset or connect to Kaggle
3. Copy code cells into notebook
4. Run cells sequentially

### Kaggle Notebooks
1. Go to https://www.kaggle.com/
2. Create new notebook
3. Add dataset: "UCI Breast Cancer Wisconsin"
4. Copy code above

### Local
```bash
pip install pandas numpy scikit-learn
python train.py
```

## Files

- `OncoMLV1.ipynb` - Complete notebook with full pipeline
- `train.py` - Training script (optional)
- Data: Download from Kaggle link above

## Requirements

```
pandas
numpy
scikit-learn
matplotlib (optional, for visualizations)
seaborn (optional, for visualizations)
```

## Why SVM?

✅ Works well with 30-dimensional data  
✅ Fast training and prediction  
✅ Linear kernel is interpretable  
✅ No GPU needed

## Results Summary

- **Training set:** 455 samples
- **Test set:** 114 samples
- **Best accuracy:** 95.61%
- **Best recall:** 95.35% (catches cancers)
- **Best specificity:** 95.77% (avoids false alarms)


## References

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). "Breast Cancer Wisconsin (Diagnostic)" UCI Machine Learning Repository.

---

**Development:** Google Colab + Kaggle Notebooks  
**Dataset:** UCI Breast Cancer Wisconsin  
**Model:** Support Vector Machine  
**Status:** Complete & Production-Ready
