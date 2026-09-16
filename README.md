# 🩺 Diabetes Prediction Using Multilayer Perceptron

A machine learning project for predicting diabetes using the **Pima Indians Diabetes Dataset**. The project performs complete data analysis, preprocessing, feature engineering, comparison of multiple machine learning models, Multilayer Perceptron (MLP) development, hyperparameter tuning, evaluation, and deployment using Streamlit.

---

## 📌 Project Overview

Diabetes is a chronic metabolic disorder that requires early detection and appropriate medical evaluation. This project develops a binary classification system to predict whether a patient is likely to have diabetes based on clinical features.

Multiple machine learning algorithms are trained and compared before developing and tuning a **Multilayer Perceptron (MLP)** neural network.

The project includes:

* Exploratory Data Analysis (EDA)
* Data quality analysis
* Physiologically implausible zero-value handling
* Median imputation
* Feature engineering
* Feature scaling
* Multiple baseline machine learning models
* MLP neural network
* Hyperparameter tuning
* ROC-AUC and Precision-Recall analysis
* Confusion matrix evaluation
* Model serialization
* Streamlit deployment

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not replace professional medical advice.

---

## 🎯 Objectives

1. Understand the structure and characteristics of the diabetes dataset.
2. Perform Exploratory Data Analysis.
3. Identify missing and physiologically implausible values.
4. Clean and preprocess the dataset.
5. Engineer meaningful features.
6. Train multiple baseline classifiers.
7. Develop a Multilayer Perceptron model.
8. Tune MLP hyperparameters.
9. Compare models using appropriate classification metrics.
10. Save the trained model and preprocessing components.
11. Develop a simple web-based prediction interface using Streamlit.

---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

### Dataset Characteristics

| Property           |   Value |
| ------------------ | ------: |
| Total Samples      |     768 |
| Predictor Features |       8 |
| Target Variable    | Outcome |
| Number of Classes  |       2 |
| Non-Diabetic       |    ~65% |
| Diabetic           |    ~35% |

### Features

| Feature                    | Description                                 |
| -------------------------- | ------------------------------------------- |
| `Pregnancies`              | Number of times pregnant                    |
| `Glucose`                  | Plasma glucose concentration                |
| `BloodPressure`            | Diastolic blood pressure                    |
| `SkinThickness`            | Triceps skinfold thickness                  |
| `Insulin`                  | 2-hour serum insulin                        |
| `BMI`                      | Body Mass Index                             |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age`                      | Patient age                                 |
| `Outcome`                  | Diabetes status: 0 or 1                     |

### Dataset Source

The dataset is available through Kaggle:

https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset

---

# 🔍 Exploratory Data Analysis

The project performs several EDA techniques:

### Univariate Analysis

* Histograms
* Box plots
* Distribution analysis

### Bivariate Analysis

* Feature distributions grouped by `Outcome`
* Box plots for important clinical features

### Correlation Analysis

A correlation heatmap is used to investigate relationships between variables.

The analysis identifies **Glucose** as one of the strongest predictors associated with the target variable.

### Outlier Analysis

The **Interquartile Range (IQR)** method is used to identify potential outliers.

Extreme but clinically plausible observations are retained rather than automatically removed.

---

# 🧹 Data Preprocessing

## Physiologically Implausible Zero Values

The following features contain zero values that are physiologically implausible:

```text
Glucose
BloodPressure
SkinThickness
Insulin
BMI
```

These zero values are treated as missing values:

```python
df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
```

### Zero-Value Summary

| Feature       | Zero Values | Percentage |
| ------------- | ----------: | ---------: |
| Glucose       |           5 |      0.65% |
| BloodPressure |          35 |      4.56% |
| SkinThickness |         227 |     29.56% |
| Insulin       |         374 |     48.70% |
| BMI           |          11 |      1.43% |

Missing values are subsequently imputed using median values.

---

# ⚙️ Feature Engineering

Additional features are created to capture nonlinear relationships and improve model performance.

### AgeGroup

Age is divided into:

```text
21–30
31–40
41–50
51+
```

### BMICategory

BMI is categorized into:

```text
Underweight
Normal
Overweight
Obese
```

### GlucoseCategory

Glucose is categorized into:

```text
Normal
Prediabetic
Diabetic Range
```

### Glucose-BMI Interaction

A combined feature is created:

```python
Glucose_BMI_interaction = Glucose * BMI
```

This attempts to capture the combined effect of glucose level and BMI.

### Insulin Log Transformation

Because insulin is strongly right-skewed:

```python
Insulin_log = np.log1p(Insulin)
```

is used to reduce skewness.

---

# 🔬 Train-Test Split

The dataset is divided using an **80:20 stratified train-test split**.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The `random_state=42` ensures reproducibility.

---

# 📏 Feature Scaling

`StandardScaler` is used to standardize the numerical features.

The scaler is fitted only on the training data:

```python
scaler.fit(X_train)
```

and then applied to both training and testing data.

This prevents information from the test set from leaking into the training process.

---

# 🤖 Machine Learning Models

The following models are evaluated:

1. Logistic Regression
2. K-Nearest Neighbors
3. Support Vector Machine
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. Multilayer Perceptron

---

# 🧠 MLP Architecture

The initial MLP architecture consists of:

```text
Input Layer
     ↓
Dense Layer — 32 neurons — ReLU
     ↓
Dropout — 0.2
     ↓
Dense Layer — 16 neurons — ReLU
     ↓
Dropout — 0.2
     ↓
Output Layer — 1 neuron — Sigmoid
```

### Configuration

| Parameter             | Value                |
| --------------------- | -------------------- |
| Hidden Layers         | 2                    |
| Hidden Units          | 32, 16               |
| Activation            | ReLU                 |
| Output Activation     | Sigmoid              |
| Optimizer             | Adam                 |
| Loss Function         | Binary Cross-Entropy |
| Dropout               | 0.2                  |
| Early Stopping        | Yes                  |
| Initial Learning Rate | 0.001                |
| Batch Size            | 32                   |

---

# 🔧 Hyperparameter Tuning

Manual grid search is performed using the following search space:

| Hyperparameter | Values                             |
| -------------- | ---------------------------------- |
| Hidden Units   | `(32,16)`, `(64,32)`, `(64,32,16)` |
| Learning Rate  | `0.001`, `0.0005`                  |
| Batch Size     | `16`, `32`                         |
| Dropout        | `0.2`, `0.3`                       |

Configurations are evaluated using validation **ROC-AUC**.

The test set is kept separate during hyperparameter tuning.

---

# 📈 Model Evaluation

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

For a diabetes screening problem, **Recall** is particularly important because false negatives can represent patients whose diabetes risk is missed.

---

# 📊 Baseline Model Results

Results obtained on the test set:

| Model                 |  Accuracy | Precision |    Recall |        F1 |   ROC-AUC |
| --------------------- | --------: | --------: | --------: | --------: | --------: |
| **Gradient Boosting** | **0.877** |     0.807 | **0.852** | **0.829** | **0.959** |
| Decision Tree         | **0.877** | **0.857** |     0.778 |     0.816 |     0.902 |
| Random Forest         |     0.870 |     0.815 |     0.815 |     0.815 |     0.950 |
| SVM                   |     0.825 |     0.737 |     0.778 |     0.757 |     0.898 |
| KNN                   |     0.792 |     0.690 |     0.741 |     0.714 |     0.869 |
| MLP                   |     0.779 |     0.672 |     0.722 |     0.696 |     0.860 |
| Logistic Regression   |     0.779 |     0.679 |     0.704 |     0.691 |     0.853 |

### Observation

Among the baseline models, **Gradient Boosting** achieved the highest ROC-AUC of **0.959** and the highest F1-score of **0.829**.

The MLP achieved:

```text
Accuracy : 0.779
Precision: 0.672
Recall   : 0.722
F1       : 0.696
ROC-AUC  : 0.860
```

The final tuned MLP should therefore be evaluated separately before making a deployment-selection claim.

---

# 💾 Model Saving

The final trained model and preprocessing components are saved as:

```text
diabetes_mlp.keras
diabetes_scaler.pkl
feature_columns.pkl
```

### Files

| File                  | Purpose                      |
| --------------------- | ---------------------------- |
| `diabetes_mlp.keras`  | Trained TensorFlow/Keras MLP |
| `diabetes_scaler.pkl` | Fitted StandardScaler        |
| `feature_columns.pkl` | Training-time feature order  |

Keeping the feature order is important because the deployment application must provide features to the model in exactly the same order used during training.

---

# 🌐 Deployment

A **Streamlit** web application is used for interactive prediction.

### Prediction Workflow

```text
User Input
    ↓
Streamlit Interface
    ↓
Feature Engineering
    ↓
Saved StandardScaler
    ↓
Saved MLP Model
    ↓
Prediction
    ↓
Diabetes Probability
```

---

# 📁 Recommended Project Structure

```text
diabetes-mlp-prediction/
│
├── data/
│   └── diabetes.csv
|
├── notebook/
│   └── diabetes_prediction.ipynb
│
├── app.py
├── diabetes_mlp.keras
├── diabetes_scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Priyansu-Rout/diabetes-mlp-prediction.git
cd diabetes-mlp-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
streamlit
joblib
```

---

# ▶️ Running the Project

### Run the notebook

Open:

```text
notebook/diabetes_prediction.ipynb
```

and execute the cells sequentially.

### Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Example Prediction

The application accepts clinical inputs such as:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

The model produces a predicted class and estimated diabetes probability.

---

# 📌 Important Considerations

Although the dataset produces promising classification results, several limitations exist:

* The dataset contains only 768 observations.
* The population is limited to Pima Indian women aged 21 and above.
* Insulin and SkinThickness contain substantial missing values represented as zeros.
* Median imputation introduces uncertainty.
* Results may not generalize to other populations.
* The model has not been clinically validated.
* A probability threshold of 0.5 may not be optimal for real-world screening.

Therefore, this project should be considered an **educational machine learning demonstration**, not a clinical diagnostic system.

---

# 🚀 Future Improvements

Possible future improvements include:

* Validation on larger and more diverse datasets.
* Cross-validation for more reliable performance estimation.
* Calibration of predicted probabilities.
* Cost-sensitive learning.
* Optimization of the classification threshold.
* SHAP-based model interpretability.
* More advanced tabular deep-learning architectures.
* Docker-based deployment.
* Cloud deployment.
* REST API deployment using FastAPI.
* Continuous model monitoring.

---

# 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**

---

# 📚 References

1. Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press, 2016.
2. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. "Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus," 1988.
3. Chollet, F. *Deep Learning with Python*, 2nd Edition, Manning Publications, 2021.

---

# 👨‍💻 Author

**Priyanshu Rout**

B.Tech — Computer Science & Engineering
Specialization: Artificial Intelligence & Machine Learning

---

## ⭐ Project Highlights

```text
✓ Complete EDA
✓ Data Cleaning
✓ Missing Value Handling
✓ Feature Engineering
✓ Feature Scaling
✓ 6 ML Baseline Models
✓ Multilayer Perceptron
✓ Hyperparameter Tuning
✓ ROC-AUC Analysis
✓ Confusion Matrix
✓ Model Serialization
✓ Streamlit Deployment
```

---

## 📄 License

This project is intended for educational and research purposes.
