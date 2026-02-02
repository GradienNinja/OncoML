# OncoML - Breast Cancer Risk Prediction

## Overview
OncoML is a machine learning system for automated breast cancer diagnosis, classifying diagnostic features as malignant or benign tumors. This project implements a Support Vector Machine (Linear Kernel) classifier achieving **95.61% accuracy** with excellent sensitivity and specificity on the UCI Breast Cancer Wisconsin dataset.

## Problem Statement
Breast cancer is one of the most prevalent and deadliest cancers worldwide. Early detection is critical for patient outcomes. Automated classification of diagnostic features from biopsy images can assist pathologists in making faster, more accurate diagnoses while reducing diagnostic variability.

## Dataset
- **Source:** [UCI Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/)
- **Size:** 569 samples total
  - Benign: 357 (62.7%)
  - Malignant: 212 (37.3%)
- **Features:** 30 diagnostic measurements derived from biopsy images
  - Radius, texture, perimeter, area, smoothness
  - Compactness, concavity, concave points
  - Symmetry, fractal dimension
  - (Computed as mean, standard error, and worst case for each feature)
- **Target:** Binary classification (Malignant = 1, Benign = 0)

## Methodology

### Preprocessing
- **Feature Scaling:** StandardScaler (zero mean, unit variance)
  - Essential for SVM distance-based algorithm
  - Applied only to training data, then to test data to prevent data leakage
- **Train-Test Split:** 80-20 stratified split
  - 455 samples for training
  - 114 samples for testing
  - Maintains class distribution to prevent bias

### Model Selection
**Algorithm:** Support Vector Machine (SVM) with Linear Kernel

Why SVM?
- Effective for high-dimensional binary classification (30 features)
- Suitable for medical diagnosis applications
- Linear kernel provides interpretability (important for clinical use)
- Robust to outliers compared to distance-based methods

### Model Configuration
```python
SVM(kernel='linear', C=1.0)
```

---

## 🎯 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **95.61%** |
| **Precision (Malignant)** | **93.18%** |
| **Recall (Malignant)** | **95.35%** |
| **F1-Score** | **94.25%** |
| **AUC-ROC** | **95.56%** |

### Clinical Interpretation

```
Confusion Matrix on Test Set (114 samples):
                  
                  Predicted Benign    Predicted Malignant
Actual Benign           68                    3
Actual Malignant         2                   41

Accuracy:    (68 + 41) / 114 = 95.61%
Sensitivity: 41 / (41 + 2) = 95.35% ← Catches 95 out of 100 actual cancers!
Specificity: 68 / (68 + 3) = 95.77% ← Correctly identifies 96 out of 100 benign cases
```

**Key Clinical Findings:**
- ✅ **Sensitivity 95.35%:** Model catches 95.35% of actual malignant tumors (only 2 missed out of 43)
- ✅ **Specificity 95.77%:** Model correctly identifies 95.77% of benign tumors (only 3 false alarms out of 71)
- ✅ **Precision 93.18%:** When model predicts malignant, it's correct 93% of the time
- ✅ **AUC-ROC 95.56%:** Excellent discrimination ability - model can distinguish between classes across all decision thresholds
- ✅ **Low False Negatives (2):** Critical for cancer detection - very few missed diagnoses
- ✅ **Reasonable False Positives (3):** Manageable false alarm rate

### Medical Significance

For cancer detection, **sensitivity (recall) is paramount:**
- **False Negative (2 cases):** Missing an actual cancer → Patient harmed
- **False Positive (3 cases):** Unnecessary further testing → Less critical, can be verified with additional tests

Our model's 95.35% sensitivity is excellent for clinical use - it catches the vast majority of malignant cases while maintaining good specificity.

---

## Installation

### Requirements
- Python 3.8+
- scikit-learn >= 1.0
- pandas
- numpy
- matplotlib
- seaborn

### Setup
```bash
# Clone repository
git clone https://github.com/GradienNinja/OncoML.git
cd OncoML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
```

---

## Usage

### Training the Model
```bash
python train.py
```

This script:
1. Loads the breast cancer dataset from Kaggle
2. Preprocesses features (scaling with StandardScaler)
3. Splits data (80-20 train-test, stratified)
4. Trains SVM with linear kernel
5. Evaluates on test set
6. Saves trained model to `models/svm_model.pkl`
7. Prints performance metrics and confusion matrix

### Making Predictions

#### Single Prediction
```python
from src.models import BreastCancerPredictor
import numpy as np

# Initialize predictor
predictor = BreastCancerPredictor('models/svm_model.pkl')

# Example diagnostic features (30 features)
features = np.array([[...]])  # 30 diagnostic measurements

# Make prediction
prediction = predictor.predict(features)
confidence = predictor.predict_proba(features)

print(f"Diagnosis: {'Malignant' if prediction[0] == 1 else 'Benign'}")
print(f"Confidence: {confidence[0]:.2%}")
```

#### Batch Predictions
```python
# Predict on multiple samples
predictions = predictor.predict_batch(X_test)
```

### Evaluating Model
```bash
python evaluate.py
```

Generates:
- Confusion matrix visualization
- ROC curve with AUC score
- Classification report (precision, recall, F1)
- Feature importance analysis

---

## Project Structure
```
OncoML/
├── data/
│   └── breast_cancer_wisconsin.csv    # Dataset (569 samples)
├── models/
│   └── svm_model.pkl                  # Trained SVM model
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Data loading & scaling
│   ├── models.py                       # Model training & prediction
│   └── evaluation.py                   # Metrics & visualization
├── notebooks/
│   └── OncoMLV1.ipynb                  # Exploratory data analysis
├── train.py                            # Training script
├── evaluate.py                         # Evaluation script
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Key Findings

### Model Strengths
✅ **High Accuracy:** 95.61% overall classification accuracy
✅ **Excellent Sensitivity:** 95.35% catch rate for malignant tumors (critical for medical use)
✅ **Good Specificity:** 95.77% correct identification of benign cases
✅ **Low False Negatives:** Only 2 missed cancers out of 43 malignant cases
✅ **High AUC-ROC:** 95.56% discrimination ability
✅ **Simple & Interpretable:** Linear kernel SVM is easy for clinicians to understand
✅ **Efficient:** Fast training and prediction time

### Model Limitations
⚠️ **Dataset Size:** 569 samples is relatively small (could benefit from more data)
⚠️ **Derived Features:** Using statistical features rather than raw image data
⚠️ **Single Split:** Only train-test split reported (k-fold cross-validation recommended)
⚠️ **Linear Kernel Only:** Could explore RBF, polynomial kernels for comparison
⚠️ **No Class Weighting:** Didn't adjust for class imbalance (could improve further)
⚠️ **Regional Variation:** Dataset from single institution (Wisconsin); generalization unknown

### Why This Approach Works
1. **Linear SVM for High-Dimensional Data:**
   - 30 features are naturally separable
   - Linear decision boundary sufficient
   - Avoids overfitting with simpler kernel

2. **Medical Context:**
   - Interpretability is crucial (doctors need to understand)
   - Linear SVM provides clear decision boundaries
   - Feature weights can be extracted for clinical insights

3. **Data Preprocessing:**
   - StandardScaler ensures equal feature contribution
   - Critical for SVM which uses distance metrics
   - Prevents high-magnitude features from dominating

---

## Validation & Reproducibility

### Data Leakage Prevention
- ✅ StandardScaler fit ONLY on training data
- ✅ Test set scaled using training set statistics
- ✅ Stratified train-test split maintains class distribution
- ✅ Fixed random state for reproducibility

### Reproducibility
```python
# Reproducible results
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Model Serialization
```python
import pickle

# Save trained model
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load for predictions
with open('models/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## Future Improvements

### Short-term (Next Steps)
- [ ] Implement k-fold cross-validation (5-10 fold) for robust performance estimation
- [ ] Test alternative kernels: RBF, Polynomial
- [ ] Hyperparameter tuning: Grid search for C parameter
- [ ] Feature selection: Identify most important diagnostic features
- [ ] Class weighting: Balance class impact in training

### Medium-term
- [ ] Compare against other algorithms:
  - Logistic Regression (baseline)
  - Random Forest
  - Gradient Boosting (XGBoost)
  - Neural Networks
- [ ] Ensemble methods: Voting, Stacking
- [ ] SHAP values for explainability
- [ ] Threshold optimization for clinical workflow
- [ ] Uncertainty quantification

### Long-term (Vision)
- [ ] Integrate actual biopsy images (CNN with transfer learning)
- [ ] Multi-class classification (tumor type, grade)
- [ ] Web interface for radiologists/pathologists
- [ ] Model explainability dashboard
- [ ] Clinical validation study
- [ ] FDA approval pathway (if applicable)
- [ ] Real-world deployment in hospitals

---

## Model Evaluation Details

### Why These Metrics Matter in Medical Context

**Accuracy (95.61%)**
- Overall correctness of model
- Useful when classes are balanced
- Should NOT be only metric for medical applications

**Precision (93.18%)**
- Of all patients flagged as malignant, how many actually have cancer?
- Higher precision = fewer false alarms
- Important to avoid unnecessary biopsies

**Recall/Sensitivity (95.35%)**
- Of all actual cancer cases, how many does model catch?
- MOST CRITICAL for cancer detection
- Missing a cancer is dangerous
- Trade-off: Can decrease precision to increase recall if needed

**Specificity (95.77%)**
- Of all benign cases, how many correctly identified?
- Prevents unnecessary treatment/worry
- Complements recall

**F1-Score (94.25%)**
- Harmonic mean of precision and recall
- Balances both metrics
- Better than accuracy for imbalanced data

**AUC-ROC (95.56%)**
- Area Under the Receiver Operating Characteristic curve
- Evaluates model's ability to distinguish classes across all thresholds
- 0.5 = random, 1.0 = perfect, our model at 0.955 is excellent

---

## Decision Threshold Optimization

Default threshold is 0.5, but can be adjusted:

```python
# Higher threshold = more conservative (fewer positives predicted)
# → Increases specificity, decreases sensitivity
# → Use if false positives are costly

# Lower threshold = more sensitive (catches more cancers)
# → Increases sensitivity, decreases specificity
# → Use if false negatives are costly (cancer detection)
```

For cancer detection, we recommend **threshold ≤ 0.5** to maximize sensitivity.

---

## References

### Dataset
Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). 
"Breast Cancer Wisconsin (Diagnostic)" UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

### Support Vector Machines
1. Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). 
   "A training algorithm for optimal margin classifiers." 
   Proceedings of the fifth annual workshop on Computational learning theory, pp. 144-152.

2. Cortes, C., & Vapnik, V. (1995). 
   "Support-vector networks." Machine Learning, 20(3), 273-297.

3. Scholkopf, B., & Smola, A. J. (2001). 
   Learning with kernels: Support vector machines, regularization, optimization, and beyond. 
   MIT press.

### Medical ML Applications
1. Esteva, A., Kuprel, B., Novoa, R. A., et al. (2019). 
   "Dermatologist-level classification of skin cancer with deep neural networks." 
   Nature, 542(7639), 115-118.

2. Rajkomar, A., Hardt, M., Howell, M. D., et al. (2018). 
   "Scalable and accurate deep learning for electronic health records." 
   npj Digital Medicine, 1(1), 18.

---

## Model Card (For Transparency)

### Model Details
- **Model Name:** OncoML SVM Classifier v1.0
- **Type:** Support Vector Machine
- **Kernel:** Linear
- **Input:** 30 diagnostic features (continuous)
- **Output:** Binary classification (Malignant=1, Benign=0)
- **Framework:** scikit-learn 1.3.0
- **Training Date:** 2024
- **Training Time:** < 1 second

### Data Characteristics
- **Training Set:** 455 samples (80%)
- **Test Set:** 114 samples (20%)
- **Class Distribution (Train):** 62.6% Benign, 37.4% Malignant
- **Class Distribution (Test):** 62.3% Benign, 37.7% Malignant

### Intended Use
- **Primary Use:** Research, educational, and demonstration purposes
- **NOT FOR:** Direct clinical decision-making without expert physician review
- **Important:** This model should assist, not replace, clinical diagnosis

### Known Limitations
- Trained on 569 samples from one institution
- Performance may vary on different populations
- Not evaluated on diverse demographic groups
- Should be validated on independent test set before clinical use

### Ethical Considerations
- Model may have demographic disparities (not evaluated)
- Recommendations: Bias audit before deployment
- Transparency: Always explain predictions to users
- Human oversight: Final diagnosis must be made by qualified physician

---

## License
MIT License - See LICENSE file for details

## Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests for new functionality
5. Commit with descriptive message (`git commit -am 'Add feature: description'`)
6. Push to branch (`git push origin feature/improvement`)
7. Open Pull Request

## Issues & Questions
- 🐛 **Bug Reports:** GitHub Issues with reproducible example
- 💡 **Feature Requests:** GitHub Issues with use case
- ❓ **Questions:** GitHub Discussions or Issues

---

## Summary Statistics

### Dataset Overview
```
Total Samples: 569
├── Benign: 357 (62.7%)
└── Malignant: 212 (37.3%)

Features: 30 diagnostic measurements
├── Mean values (10 features)
├── Standard error (10 features)
└── Worst/largest values (10 features)

Train-Test Split: 80-20
├── Training: 455 samples
└── Testing: 114 samples
```

### Model Performance Summary
```
┌─────────────────────────┐
│  OncoML Performance     │
├─────────────────────────┤
│ Accuracy:      95.61%   │
│ Precision:     93.18%   │
│ Recall:        95.35%   │
│ Specificity:   95.77%   │
│ F1-Score:      94.25%   │
│ AUC-ROC:       95.56%   │
│                         │
│ False Negatives: 2      │
│ False Positives: 3      │
│ True Negatives: 68      │
│ True Positives: 41      │
└─────────────────────────┘
```

---

## Getting Started Quickly

```bash
# 1. Clone and setup
git clone https://github.com/GradienNinja/OncoML.git
cd OncoML
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train model
python train.py

# 3. Evaluate model
python evaluate.py

# 4. Make predictions
python -c "from src.models import BreastCancerPredictor; predictor = BreastCancerPredictor(); print(predictor.predict([...]))"
```

---

## Citation
If you use this project in research, please cite:

```bibtex
@software{oncoml2024,
  title={OncoML: Breast Cancer Risk Prediction using Support Vector Machines},
  author={GradienNinja},
  year={2024},
  url={https://github.com/GradienNinja/OncoML}
}
```

---

**Last Updated:** February 2024
**Version:** 1.0
**Status:** Stable - Ready for research use
