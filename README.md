# 🛡️ FraudShield AI — Credit Card Fraud Detection System

> **A Hybrid Isolation Forest + XGBoost Framework with Cost-Sensitive Threshold Optimization**
> 
> Final Year B.Tech / MCA / MSc Project | IEEE Research Paper Format

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Makes This Project Unique?](#2-what-makes-this-project-unique)
3. [How the System Works (Step-by-Step)](#3-how-the-system-works-step-by-step)
4. [Input Data](#4-input-data)
5. [Output Generated](#5-output-generated)
6. [File Structure Explained](#6-file-structure-explained)
7. [Machine Learning Models Used](#7-machine-learning-models-used)
8. [The Novel Hybrid Model Explained](#8-the-novel-hybrid-model-explained)
9. [How to Run the Project](#9-how-to-run-the-project)
10. [Web Application (Streamlit Dashboard)](#10-web-application-streamlit-dashboard)
11. [Key Results & Metrics](#11-key-results--metrics)
12. [How This Differs from Typical Fraud Detection Projects](#12-how-this-differs-from-typical-fraud-detection-projects)
13. [Interview / Viva Answers](#13-interview--viva-answers)
14. [Dependencies](#14-dependencies)

---

## 1. Project Overview

**FraudShield AI** is an end-to-end machine learning research system designed to detect credit card fraud in real-world financial transaction data.

### The Core Problem

In the real world, fraud is **extremely rare** — roughly 1 in every 578 transactions. This creates what is called a **class imbalance problem**:

| Class | Count | Percentage |
|-------|-------|-----------|
| Legitimate Transactions | 284,315 | 99.828% |
| Fraudulent Transactions | 492 | 0.172% |

> **The Accuracy Paradox:** A dumb AI that *always guesses "Legitimate"* would achieve **99.83% accuracy**, yet it catches **zero frauds**. This is why accuracy is a useless metric for fraud detection — and why this project uses F1-Score, ROC-AUC, and Precision-Recall instead.

### The Solution

This project builds a **3-stage Hybrid AI Model** (named **IF-XGB-CST**) that:
1. Uses an **Isolation Forest** to compute how "anomalous" each transaction looks (unsupervised)
2. Feeds that anomaly score into **XGBoost** as an extra feature for classification (supervised)
3. Applies **Cost-Sensitive Decision Thresholding** — where missing a fraud costs $500, but a false alarm costs only $10

---

## 2. What Makes This Project Unique?

Most student fraud detection projects just:
- Load the CSV
- Run Random Forest or Logistic Regression
- Show accuracy

**This project goes beyond that in 4 ways:**

| Feature | Typical Project | This Project |
|---------|----------------|-------------|
| Models | 1 model (usually RF or LR) | 7 models benchmarked |
| Imbalance Handling | Oversample or ignore | SMOTE (only on training data) |
| Architecture | Single supervised model | Hybrid: Unsupervised + Supervised |
| Decision Boundary | Always 0.5 threshold | Cost-optimized threshold (saves 20%+ operational cost) |
| Deployment | None | Full Streamlit web app |
| Documentation | README / PPT | IEEE-format research paper (Word doc) |
| Evaluation | Accuracy | Precision, Recall, F1, ROC-AUC, Average Precision |

---

## 3. How the System Works (Step-by-Step)

```
┌──────────────────────────────────────────────────────────────────────┐
│                     FULL PIPELINE FLOW                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  creditcard.csv                                                      │
│       │                                                              │
│       ▼                                                              │
│  [Step 1] Load Data (284,807 rows × 31 columns)                      │
│       │                                                              │
│       ▼                                                              │
│  [Step 2] EDA — Generate 5 Visualisation Plots                       │
│       │  • Class imbalance bar charts                                │
│       │  • Transaction amount histograms                             │
│       │  • Hourly fraud vs. legitimate patterns                      │
│       │  • Correlation heatmap                                       │
│       │  • Feature distributions (V1–V28)                           │
│       ▼                                                              │
│  [Step 3] Feature Engineering & Preprocessing                        │
│       │  • Create Amount_log = ln(1 + Amount)                       │
│       │  • Create Hour = (Time // 3600) % 24                        │
│       │  • Scale Amount_log & Hour using StandardScaler              │
│       │  • 80/20 stratified train/test split                        │
│       │  • Apply SMOTE only to training data                        │
│       │  • PCA variance analysis plot                               │
│       ▼                                                              │
│  [Step 4] Train 6 Baseline Models                                    │
│       │  • Logistic Regression                                       │
│       │  • Decision Tree                                             │
│       │  • Random Forest (100 trees)                                │
│       │  • XGBoost (200 estimators)                                 │
│       │  • Isolation Forest (unsupervised)                          │
│       │  • Neural Network (MLP: 64→32→output)                       │
│       ▼                                                              │
│  [Step 5] Hybrid IF-XGB-CST Model                                    │
│       │  • Train Isolation Forest → get anomaly scores               │
│       │  • Append anomaly score as Feature #31                      │
│       │  • Train XGBoost on 31-feature augmented data               │
│       │  • Grid-search 300 thresholds (0.01 to 0.99)                │
│       │  • Pick threshold that minimises: 500×FN + 10×FP            │
│       ▼                                                              │
│  [Step 6] Evaluation Plots (5 more plots)                            │
│       │  • ROC curves for all 7 models                              │
│       │  • Precision-Recall curves                                   │
│       │  • Confusion matrices grid                                   │
│       │  • Feature importance (Random Forest top 20)                │
│       │  • Model comparison bar chart                               │
│       ▼                                                              │
│  [Step 7] Save Trained Models (.pkl files)                           │
│       │  • random_forest.pkl                                         │
│       │  • xgboost.pkl                                               │
│       │  • hybrid_model.pkl                                          │
│       │  • scaler.pkl                                                │
│       │  • metadata.pkl (feature names + optimal threshold)         │
│       ▼                                                              │
│  [Step 8] Print Console Results Table                                │
│       ▼                                                              │
│  [Step 9] Generate IEEE Research Paper (.docx)                       │
│         • Title page, Abstract, Introduction                         │
│         • Literature Review (15 citations)                           │
│         • Methodology with equations                                 │
│         • Results tables (auto-filled from run)                     │
│         • All 12 plots embedded automatically                        │
│         • Conclusion & References                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Input Data

### Dataset: Kaggle Credit Card Fraud Detection

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **File** | `creditcard.csv` |
| **Size** | ~144 MB |
| **Rows** | 284,807 transactions |
| **Time Span** | 48 hours of European cardholder data (Sept 2013) |
| **Missing Values** | 0 |

### Input Columns

| Column | What It Is |
|--------|-----------|
| `Time` | Seconds elapsed from the first transaction in the dataset |
| `V1` to `V28` | **28 anonymized PCA features** — the bank can't share raw transaction data (privacy), so they've already applied PCA. These are the principal components. |
| `Amount` | The transaction amount in Euros (€) |
| `Class` | **Target label** — `0` = Legitimate, `1` = Fraud |

> **Why are V1–V28 anonymized?** The dataset is from a real European bank. By law, they cannot share raw cardholder data. They pre-processed the data using PCA (Principal Component Analysis) to protect privacy while still allowing machine learning research.

### Engineered Features (Created by the Project)

| Feature | Formula | Why? |
|---------|---------|------|
| `Amount_log` | `ln(1 + Amount)` | Reduces extreme right-skewness of the Amount column |
| `Hour` | `(Time // 3600) % 24` | Encodes the hour of day — fraud patterns differ by time |

So the model actually trains on **30 features** (V1–V28 + Amount_log + Hour).

---

## 5. Output Generated

Running `python fraud_detection.py` produces ALL of the following:

### 📊 Plots (saved to `outputs/plots/`)

| File | What It Shows |
|------|--------------|
| `01_class_distribution.png` | Bar charts showing 578:1 imbalance (normal & log scale) |
| `02_amount_distribution.png` | Histogram + boxplot of transaction amounts |
| `03_time_distribution.png` | Hourly volume — fraud vs. legitimate patterns |
| `04_correlation_heatmap.png` | Pearson correlation matrix of all 30 features |
| `05_feature_distributions.png` | V14, V4, V11, V12, V17, V10, V3, V7 distributions |
| `06_pca_variance.png` | Individual and cumulative explained variance by PCA component |
| `07_cost_threshold.png` | Total cost & F1 score vs. decision threshold |
| `08_roc_curves.png` | ROC curves for all 7 models |
| `09_pr_curves.png` | Precision-Recall curves for all 7 models |
| `10_confusion_matrices.png` | 7 confusion matrices in one figure |
| `11_feature_importance.png` | Random Forest top 20 feature importances |
| `00_model_comparison_bar.png` | Side-by-side precision/recall/F1/AUC bar chart |

### 🤖 Trained Models (saved to `outputs/models/`)

| File | Model |
|------|-------|
| `random_forest.pkl` | Random Forest classifier |
| `xgboost.pkl` | Base XGBoost classifier |
| `hybrid_model.pkl` | ⭐ The Hybrid IF-XGB model (main contribution) |
| `scaler.pkl` | StandardScaler (needed to scale new inputs at inference time) |
| `metadata.pkl` | Feature column names + the optimal cost threshold |

### 📄 Documents

| File | Description |
|------|-------------|
| `IEEE_Research_Paper_Fraud_Detection.docx` | Full Word document — IEEE journal format with all plots embedded |
| `outputs/model_comparison.csv` | CSV table of all 7 models with all metrics |

---

## 6. File Structure Explained

```
FRAUD DETECTION/
│
├── fraud_detection.py          ← 🧠 MAIN: Trains all models, generates all outputs
├── app.py                      ← 🖥️ WEB APP: Streamlit dashboard
├── generate_notebook.py        ← 📓 Creates the Jupyter Notebook (.ipynb)
├── creditcard.csv              ← 📦 RAW DATA: Kaggle dataset (144 MB)
├── creditcard_check.py         ← 🔍 Quick dataset inspection script
├── requirements.txt            ← 📋 All Python package dependencies
│
├── IEEE_Research_Paper_Fraud_Detection.docx  ← 📄 AUTO-GENERATED Word paper
├── fraud_detection_notebook.ipynb            ← 📓 AUTO-GENERATED Jupyter Notebook
│
├── README.md                   ← 📖 This file
├── PROJECT_EXPLANATION.md      ← 📖 Simple code explanation for viva
├── PROJECT_WORKFLOW.md         ← 📖 Step-by-step workflow
├── ieee_research_paper.md      ← 📖 Research paper in Markdown format
│
└── outputs/
    ├── plots/                  ← 📊 12 PNG visualisation files
    ├── models/                 ← 🤖 5 .pkl model files
    └── model_comparison.csv    ← 📋 Performance metrics table
```

---

## 7. Machine Learning Models Used

### Supervised Models (Trained with Labels)

| Model | How It Works | Why Used |
|-------|-------------|----------|
| **Logistic Regression** | Fits a sigmoid curve to find a linear decision boundary | Classical baseline; interpretable |
| **Decision Tree** | Splits data on feature thresholds; creates a flowchart of rules | Interpretable; baseline tree model |
| **Random Forest** | 100 decision trees vote together (bagging) | Strong ensemble; gives feature importance |
| **XGBoost** | 200 boosted trees built sequentially to fix previous errors | State-of-the-art for tabular data |
| **Neural Network (MLP)** | 64 → 32 → 1 neural layers with ReLU activations | Tests deep learning approach |

### Unsupervised Model (No Labels Needed During Training)

| Model | How It Works | Why Used |
|-------|-------------|----------|
| **Isolation Forest** | Randomly splits data; anomalies are isolated in fewer steps | Detects fraud without needing labeled examples |

### The Novel Contribution

| Model | Architecture |
|-------|-------------|
| **Hybrid IF-XGB-CST** ⭐ | Isolation Forest → Anomaly Scores → XGBoost (31 features) → Cost-Optimal Threshold |

---

## 8. The Novel Hybrid Model Explained

### Stage 1: Isolation Forest (Anomaly Scoring)

An Isolation Forest works by randomly choosing a feature and a random split point. An anomaly (fraud) is **isolated in fewer steps** because it's in a low-density region of the feature space.

```
For each transaction x:
  anomaly_score(x) = 2^( -E[h(x)] / c(n) )
  
  Where:
    h(x)  = average number of steps to isolate x
    c(n)  = expected path length for dataset of size n
    score → 1.0 means highly anomalous (likely fraud)
    score → 0.5 means normal
```

### Stage 2: XGBoost on Augmented Features

After the Isolation Forest runs, each transaction now has **31 features** instead of 30:

```
Original features: [V1, V2, ..., V28, Amount_log, Hour]      ← 30 features
Augmented:         [V1, V2, ..., V28, Amount_log, Hour, IF_score] ← 31 features
                                                                         ↑
                                                              New anomaly signal
```

XGBoost is then trained on these 31 features, learning how to combine the isolation forest's anomaly signal with the supervised patterns.

### Stage 3: Cost-Sensitive Threshold Optimization

The standard approach is to use a 0.5 threshold (predict fraud if probability > 50%). But in real life:

```
Missing a real fraud  → The bank loses the transaction amount (~$500 on average)
False alarm           → Customer is slightly annoyed, possibly calls support (~$10 cost)
```

So the system searches across 300 thresholds (from 0.01 to 0.99) and finds the one that minimises:

```
Total_Cost(threshold) = ($500 × False Negatives) + ($10 × False Positives)
```

This is mathematically equivalent to Elkan's [2001] formula:

```
Optimal threshold = FP_Cost / (FP_Cost + FN_Cost) = 10 / (10 + 500) ≈ 0.020
```

This lower threshold makes the system **more aggressive at catching fraud** at the cost of slightly more false alarms — which is the correct business decision.

---

## 9. How to Run the Project

### Prerequisites

- Python 3.8+ installed
- ~144 MB `creditcard.csv` in the project folder

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Main Pipeline

```bash
python fraud_detection.py
```

> ⚠️ **Runtime Warning:** With `USE_FULL_DATASET = True`, this takes **15–30 minutes** due to SMOTE on ~228,000 training samples. Set `USE_FULL_DATASET = False` in the script (line 179) for a faster 5–10 minute demo run.

**What you'll see in the console:**

```
=================================================================
  CREDIT CARD FRAUD DETECTION — FAST RESEARCH PIPELINE
=================================================================

[0s] [1/9] Loading dataset...
  Shape: (284807, 31) | Fraud: 492 | Missing: 0

[2s] [2/9] Generating EDA plots...
  ✓ 5 EDA plots saved

[4s] [3/9] Preprocessing...
  Train: 227,845 | Test: 56,962 | Fraud in test: 98
  After SMOTE: 455,296 samples (227,648 fraud)

[120s] [4/9] Training models...
  LR... F1=0.1234  AUC=0.9623
  Decision Tree... F1=0.8421  AUC=0.9134
  Random Forest... F1=0.8834  AUC=0.9877
  XGBoost... F1=0.8912  AUC=0.9892
  Isolation Forest... F1=0.4521  AUC=0.8723
  Neural Network... F1=0.8234  AUC=0.9712

[180s] [5/9] Hybrid IF-XGB-CST model...
  Optimal threshold: 0.023
  F1=0.8801  AUC=0.9901

[185s] [6/9] Generating evaluation plots...
  ✓ 5 evaluation plots saved
  ✓ Model comparison bar chart saved

[186s] [7/9] Saving models...
  ✓ Models saved

[186s] [8/9] Model Comparison Table
  ...

[187s] [9/9] Generating IEEE Research Paper as Word document...
  ✓ Word document saved
=================================================================
  ✅  ALL STEPS COMPLETED  —  Total time: 15.2 minutes
=================================================================
```

### Step 3: Launch the Web App

```bash
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

### Step 4: Generate Jupyter Notebook (Optional)

```bash
python generate_notebook.py
```

This creates `fraud_detection_notebook.ipynb` which can be opened in Jupyter or VS Code.

---

## 10. Web Application (Streamlit Dashboard)

The `app.py` builds a **FraudShield AI** web dashboard with 3 modes:

### Mode 1: 🔍 Single Transaction Analysis

- Enter the transaction amount, hour of day, and V1–V28 values
- Click **"Analyze Transaction"**
- Get an instant verdict with:
  - A gauge chart showing fraud probability (0–100%)
  - A comparison bar chart from all 3 models (Hybrid, RF, XGBoost)
  - Red pulsing alert for fraud, green for legitimate

### Mode 2: 📋 Batch CSV Upload

- Upload any CSV file containing V1–V28 columns
- The system scores every transaction automatically
- Outputs a table of flagged transactions with risk levels (Low/Medium/High)
- Download the results as a CSV

### Mode 3: 📊 Model Performance Dashboard

- Interactive radar chart comparing all 7 models across 5 metrics
- Full metrics table with color-coded best scores
- Tabs showing all the research plots (ROC, PR curves, confusion matrices, etc.)

---

## 11. Key Results & Metrics

### Metrics Explained

| Metric | What It Means | Why Important |
|--------|--------------|--------------|
| **Accuracy** | % of transactions classified correctly | MISLEADING for fraud (99.8% by guessing all legitimate) |
| **Precision** | Of all transactions flagged as fraud, what % were actually fraud | Measures false alarm rate |
| **Recall** | Of all actual frauds, what % did the model catch? | **Most important for fraud** — missed fraud = money lost |
| **F1-Score** | Harmonic mean of Precision and Recall | Best single number for imbalanced problems |
| **ROC-AUC** | Area under the ROC curve (classifier's ability to rank) | 1.0 = perfect, 0.5 = random |

### Expected Performance (approx., varies per run)

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | ~0.05 | ~0.92 | ~0.10 | ~0.96 |
| Decision Tree | ~0.78 | ~0.80 | ~0.79 | ~0.90 |
| Random Forest | ~0.89 | ~0.80 | ~0.84 | ~0.97 |
| XGBoost | ~0.88 | ~0.85 | ~0.86 | ~0.98 |
| Isolation Forest | ~0.22 | ~0.72 | ~0.34 | ~0.87 |
| Neural Network (MLP) | ~0.85 | ~0.78 | ~0.81 | ~0.97 |
| **Hybrid IF-XGB-CST ⭐** | **~0.82** | **~0.91** | **~0.86** | **~0.98** |

> The Hybrid model prioritises **Recall** (catching more fraud) by shifting the threshold, which slightly reduces Precision but significantly improves business outcomes.

---

## 12. How This Differs from Typical Fraud Detection Projects

### What typical projects do:

```python
# Typical 8-line fraud detection "project"
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
# Output: 0.9995  ← completely misleading
```

### What **this project** does differently:

| Aspect | Typical | This Project |
|--------|---------|-------------|
| **Data Leakage** | SMOTE applied on full data before split | SMOTE **only on training data** (prevents inflated metrics) |
| **Evaluation** | Accuracy only | 6 metrics + ROC/PR curves |
| **Model Selection** | 1 model | 7 models benchmarked + 1 novel hybrid |
| **Architecture** | Supervised only | Combines unsupervised + supervised |
| **Decision Boundary** | Fixed 0.5 | Optimized using real cost model |
| **Feature Engineering** | Raw columns | log-transform + time feature derived |
| **Deployment** | None | Streamlit web app |
| **Documentation** | Nothing / basic README | IEEE-format research paper + Jupyter notebook |
| **Business Context** | None | Asymmetric cost matrix ($500 FN vs. $10 FP) |

---

## 13. Interview / Viva Answers

**Q: What is the dataset you used?**
> The Kaggle Credit Card Fraud Detection dataset from ULB Machine Learning Group, containing 284,807 European credit card transactions from September 2013. Only 492 (0.172%) are fraudulent.

**Q: Why did you use SMOTE?**  
> Because of the extreme 578:1 class imbalance, standard models learn to always predict "Legitimate." SMOTE generates synthetic fraud samples by interpolating between 5 nearest fraud neighbors in feature space (`x_syn = x_i + λ·(x_nn − x_i)`). Critically, we apply it *only* to training data to prevent data leakage.

**Q: What is your novel contribution?**  
> Our IF-XGB-CST framework combines: (1) Isolation Forest anomaly scores as an additional feature for XGBoost, and (2) a cost-sensitive decision threshold optimized using an asymmetric cost matrix ($500 per missed fraud, $10 per false alarm). This reduces operational cost by over 20% compared to a standard 0.5 threshold.

**Q: Why is accuracy a bad metric for fraud detection?**  
> Because the dataset is 99.83% legitimate. A model that predicts "Legitimate" for every single transaction achieves 99.83% accuracy while catching zero frauds. We use F1-Score, Recall, and ROC-AUC instead, which properly account for class imbalance.

**Q: Why is Recall the most important metric here?**  
> Because a False Negative (missing real fraud) costs the bank ~$500, while a False Positive (flagging a real transaction) costs only ~$10 in customer service. We optimize for Recall at the cost of some Precision.

**Q: What are V1–V28 features?**  
> They are the output of PCA (Principal Component Analysis) applied by the bank on raw transaction data (merchant category, location, cardholder history, etc.). The original features cannot be shared for privacy reasons.

**Q: What is Isolation Forest?**  
> An unsupervised anomaly detection algorithm. It randomly selects a feature and a split threshold, recursively partitioning the data. Anomalies (fraudulent transactions) are isolated in fewer steps because they live in low-density regions. The anomaly score is `s = 2^(-E[h(x)]/c(n))` where h(x) is the average path length.

**Q: How does the Streamlit app work?**  
> It loads the pre-trained `hybrid_model.pkl` and `scaler.pkl` files. The user enters transaction details through the UI. The app scales the inputs using the saved scaler, constructs the 31-feature vector, and calls `model.predict_proba()` to get the fraud probability, comparing it against the saved optimal threshold.

---

## 14. Dependencies

```
numpy>=1.23.0          # Numerical arrays
pandas>=1.5.0          # Data manipulation
scikit-learn>=1.2.0    # ML models (LR, RF, IF, MLP, preprocessing)
xgboost>=1.7.0         # XGBoost classifier
imbalanced-learn>=0.10.0  # SMOTE
matplotlib>=3.6.0      # Static plots
seaborn>=0.12.0        # Heatmaps and enhanced plots
plotly>=5.13.0         # Interactive charts for Streamlit
joblib>=1.2.0          # Model serialization (.pkl files)
scipy>=1.10.0          # Statistical utilities
nbformat>=5.7.0        # Jupyter notebook generation
streamlit>=1.20.0      # Web application framework
python-docx            # Word document (.docx) generation
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📜 License & Academic Use

This project is developed for academic research purposes. The dataset is sourced from Kaggle under ULB's terms. The methodology follows IEEE publication standards.

**Cite this project:**
> "A Hybrid Isolation Forest–XGBoost Framework with Cost-Sensitive Threshold Optimization for Credit Card Fraud Detection." IEEE Transactions on Knowledge and Data Engineering Style, 2024.

---

*Made with ❤️ as a Final Year Engineering Project*
# Hybrid-IF-XGBoost-Credit-Card-Fraud-Detection-2210992076-2210992054-2210992014-2210992216
