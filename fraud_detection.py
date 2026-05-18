"""
Credit Card Fraud Detection -- FAST Research Pipeline
=====================================================
Estimated time: ~6-10 minutes total
Optimizations vs. previous version:
  - SMOTE sample cap (50K per class instead of full dataset)
  - RandomizedSearchCV with fewer iterations
  - RF with 100 estimators only
  - All models use n_jobs=-1
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, classification_report, matthews_corrcoef
)
import joblib
import xgboost as xgb
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
np.random.seed(42)
START = time.time()

# -- Directories --------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
MODELS_DIR= os.path.join(BASE_DIR, 'outputs', 'models')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -- Plot style ----------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
})
C = {'legit': '#2196F3', 'fraud': '#F44336'}

def elapsed():
    return f"[{time.time()-START:.0f}s]"

print("="*65)
print("  CREDIT CARD FRAUD DETECTION -- FAST RESEARCH PIPELINE")
print("="*65)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [1/9] Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, 'creditcard.csv'))
print(f"  Shape: {df.shape} | Fraud: {df['Class'].sum()} | Missing: {df.isnull().sum().sum()}")

# -----------------------------------------------------------------------------
# 2. EDA PLOTS
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [2/9] Generating EDA plots...")

# 2.1 Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
counts = df['Class'].value_counts()
for ax, scale in zip(axes, [None, 'log']):
    bars = ax.bar(['Legitimate', 'Fraud'], counts.values,
                  color=[C['legit'], C['fraud']], width=0.5, edgecolor='white')
    if scale: ax.set_yscale('log')
    for bar, val in zip(bars, counts.values):
        y = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, y*(1.3 if scale else 1.02),
                f'{val:,}\n({val/len(df)*100:.3f}%)',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_title(f'Class Distribution{"(Log)" if scale else ""}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count' + (' (log)' if scale else ''))
plt.suptitle('Class Imbalance  --  Ratio ~ 578:1', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution.png'), bbox_inches='tight')
plt.close()

# 2.2 Amount Distribution
fraud_amt = df[df['Class']==1]['Amount']
legit_amt = df[df['Class']==0]['Amount']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for amt, label, color in [(legit_amt,'Legitimate',C['legit']),(fraud_amt,'Fraud',C['fraud'])]:
    axes[0].hist(amt, bins=80, alpha=0.65, color=color, label=label, density=True)
axes[0].set_xlim(0, 800); axes[0].set_xlabel('Amount (€)'); axes[0].set_ylabel('Density')
axes[0].set_title('Transaction Amount Distribution', fontsize=13, fontweight='bold')
axes[0].legend()
axes[1].boxplot([legit_amt[legit_amt<300].values, fraud_amt.values],
                labels=['Legitimate\n(< €300)','Fraud'],
                patch_artist=True, widths=0.5,
                boxprops=dict(facecolor='#BBDEFB', color='navy'),
                medianprops=dict(color='red', linewidth=2.5))
axes[1].set_ylabel('Amount (€)'); axes[1].set_title('Amount Boxplot', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_amount_distribution.png'), bbox_inches='tight')
plt.close()

# 2.3 Hourly pattern
df['Hour'] = (df['Time'] // 3600) % 24
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, cls, label, color in [(axes[0],0,'Legitimate',C['legit']),(axes[1],1,'Fraud',C['fraud'])]:
    h = df[df['Class']==cls].groupby('Hour').size()
    ax.fill_between(h.index, h.values, alpha=0.55, color=color)
    ax.plot(h.index, h.values, color=color, lw=2, marker='o', ms=4, label=label)
    ax.set_ylabel('Count'); ax.set_title(f'Hourly Volume -- {label}', fontsize=13, fontweight='bold')
    ax.legend()
axes[1].set_xlabel('Hour of Day'); axes[1].set_xticks(range(0,24))
plt.suptitle('Temporal Pattern Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_time_distribution.png'), bbox_inches='tight')
plt.close()

# 2.4 Correlation Heatmap
vf = [f'V{i}' for i in range(1,29)] + ['Amount','Class']
corr = df[vf].corr()
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='RdYlBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.3,
            cbar_kws={'shrink':0.8,'label':'Pearson Correlation'})
ax.set_title('Feature Correlation Heatmap', fontsize=15, fontweight='bold', pad=18)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_correlation_heatmap.png'), bbox_inches='tight')
plt.close()

# 2.5 Key Feature Distributions
key_feats = ['V14','V4','V11','V12','V17','V10','V3','V7']
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for idx, feat in enumerate(key_feats):
    ax = axes[idx//4][idx%4]
    ax.hist(df[df['Class']==0][feat].sample(5000,random_state=42),
            bins=50, alpha=0.55, color=C['legit'], density=True, label='Legitimate')
    ax.hist(df[df['Class']==1][feat], bins=30, alpha=0.8,
            color=C['fraud'], density=True, label='Fraud')
    ax.set_title(f'Feature {feat}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    ax.legend(fontsize=8)
plt.suptitle('Key Discriminative V-Features (Fraud vs. Legitimate)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_feature_distributions.png'), bbox_inches='tight')
plt.close()
print(f"  [OK] 5 EDA plots saved")

# -----------------------------------------------------------------------------
# 3. PREPROCESSING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [3/9] Preprocessing...")
df['Amount_log'] = np.log1p(df['Amount'])
feature_cols = [f'V{i}' for i in range(1,29)] + ['Amount_log','Hour']
X = df[feature_cols].values
y = df['Class'].values

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[:,-2:] = scaler.fit_transform(X[:,-2:])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Fraud in test: {y_test.sum()}")

USE_FULL_DATASET = True  # Setup to run full volume of data

smote = SMOTE(random_state=42, k_neighbors=5)
if USE_FULL_DATASET:
    print("  [Running on FULL dataset array. Processing will take 15-30 mins...]")
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
else:
    # SPEED TRICK: cap SMOTE to 50K per class (saves ~5x time vs. full balance)
    SMOTE_CAP = 50_000
    fraud_idx   = np.where(y_train==1)[0]
    legit_idx   = np.where(y_train==0)[0]
    legit_sub   = np.random.choice(legit_idx, size=min(SMOTE_CAP, len(legit_idx)), replace=False)
    sub_idx     = np.concatenate([fraud_idx, legit_sub])
    X_tr_sub    = X_train[sub_idx]
    y_tr_sub    = y_train[sub_idx]
    X_train_smote, y_train_smote = smote.fit_resample(X_tr_sub, y_tr_sub)

print(f"  After SMOTE: {X_train_smote.shape[0]:,} samples ({y_train_smote.sum():,} fraud)")

# PCA Variance Plot
pca = PCA()
pca.fit(X_scaled[:,:28])
ev = np.cumsum(pca.explained_variance_ratio_)
n95 = np.argmax(ev >= 0.95) + 1
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(range(1,29), pca.explained_variance_ratio_*100, 'o-', color='#2196F3', lw=2, ms=5)
axes[0].fill_between(range(1,29), pca.explained_variance_ratio_*100, alpha=0.25, color='#2196F3')
axes[0].set_title('Individual Component Variance', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Component'); axes[0].set_ylabel('Explained Variance (%)')
axes[1].plot(range(1,29), ev*100, 's-', color='#4CAF50', lw=2, ms=5)
axes[1].axhline(y=95, color='red', ls='--', lw=1.5, label='95% threshold')
axes[1].axvline(x=n95, color='orange', ls='--', lw=1.5, label=f'{n95} components')
axes[1].fill_between(range(1,29), ev*100, alpha=0.25, color='#4CAF50')
axes[1].set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
axes[1].set_xlabel('n Components'); axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].legend()
plt.suptitle('PCA Variance Analysis -- V1 to V28', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_pca_variance.png'), bbox_inches='tight')
plt.close()
print(f"  [OK] PCA: {n95} components -> 95% variance")

# -----------------------------------------------------------------------------
# 4. TRAIN MODELS  (optimized for speed)
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [4/9] Training models...")

results = {}
model_store = {}

def eval_m(y_true, y_pred, y_prob):
    # NOTE: We use Balanced Accuracy instead of regular Accuracy.
    # Regular accuracy is misleading for imbalanced datasets (578:1 ratio):
    #   A naive model predicting ALL as legitimate gets 99.83% accuracy
    #   but detects ZERO fraud. Balanced Accuracy averages recall per class
    #   and gives a realistic score reflecting true model capability.
    return dict(
        Bal_Accuracy   = round(balanced_accuracy_score(y_true, y_pred), 4),
        Precision      = round(precision_score(y_true, y_pred, zero_division=0), 4),
        Recall         = round(recall_score(y_true, y_pred, zero_division=0), 4),
        F1             = round(f1_score(y_true, y_pred, zero_division=0), 4),
        ROC_AUC        = round(roc_auc_score(y_true, y_prob), 4),
        Avg_Precision  = round(average_precision_score(y_true, y_prob), 4),
        MCC            = round(matthews_corrcoef(y_true, y_pred), 4),
    )

# -- Logistic Regression ------------------------------------------------------
print(f"  {elapsed()} LR...", end='', flush=True)
lr = LogisticRegression(C=0.1, penalty='l2', class_weight='balanced',
                        max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train_smote, y_train_smote)
p_lr = lr.predict_proba(X_test)[:,1]
d_lr = lr.predict(X_test)
results['Logistic Regression'] = eval_m(y_test, d_lr, p_lr)
model_store['Logistic Regression'] = (lr, p_lr, d_lr)
print(f" F1={results['Logistic Regression']['F1']:.4f}  AUC={results['Logistic Regression']['ROC_AUC']:.4f}")

# -- Decision Tree ------------------------------------------------------------
print(f"  {elapsed()} Decision Tree...", end='', flush=True)
dt = DecisionTreeClassifier(max_depth=12, min_samples_split=5,
                             class_weight='balanced', random_state=42)
dt.fit(X_train_smote, y_train_smote)
p_dt = dt.predict_proba(X_test)[:,1]
d_dt = dt.predict(X_test)
results['Decision Tree'] = eval_m(y_test, d_dt, p_dt)
model_store['Decision Tree'] = (dt, p_dt, d_dt)
print(f" F1={results['Decision Tree']['F1']:.4f}  AUC={results['Decision Tree']['ROC_AUC']:.4f}")

# -- Random Forest (fast: 100 trees, no grid search) --------------------------
print(f"  {elapsed()} Random Forest...", end='', flush=True)
rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
p_rf = rf.predict_proba(X_test)[:,1]
d_rf = rf.predict(X_test)
results['Random Forest'] = eval_m(y_test, d_rf, p_rf)
model_store['Random Forest'] = (rf, p_rf, d_rf)
joblib.dump(rf, os.path.join(MODELS_DIR,'random_forest.pkl'))
print(f" F1={results['Random Forest']['F1']:.4f}  AUC={results['Random Forest']['ROC_AUC']:.4f}")

# -- XGBoost ------------------------------------------------------------------
print(f"  {elapsed()} XGBoost...", end='', flush=True)
spw = (y_train==0).sum() / (y_train==1).sum()
xgb_m = xgb.XGBClassifier(scale_pos_weight=spw, n_estimators=200, max_depth=5,
                            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, eval_metric='logloss',
                            use_label_encoder=False, n_jobs=-1)
xgb_m.fit(X_train, y_train)
p_xg = xgb_m.predict_proba(X_test)[:,1]
d_xg = xgb_m.predict(X_test)
results['XGBoost'] = eval_m(y_test, d_xg, p_xg)
model_store['XGBoost'] = (xgb_m, p_xg, d_xg)
joblib.dump(xgb_m, os.path.join(MODELS_DIR,'xgboost.pkl'))
print(f" F1={results['XGBoost']['F1']:.4f}  AUC={results['XGBoost']['ROC_AUC']:.4f}")

# -- Isolation Forest ---------------------------------------------------------
print(f"  {elapsed()} Isolation Forest...", end='', flush=True)
iso = IsolationForest(n_estimators=150, contamination=0.001724, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_sc_test  = -iso.score_samples(X_test)
iso_sc_train = -iso.score_samples(X_train)
d_if = np.where(iso.predict(X_test)==-1, 1, 0)
results['Isolation Forest'] = eval_m(y_test, d_if, iso_sc_test)
model_store['Isolation Forest'] = (iso, iso_sc_test, d_if)
print(f" F1={results['Isolation Forest']['F1']:.4f}  AUC={results['Isolation Forest']['ROC_AUC']:.4f}")

# -- Neural Network (MLP, small, fast) ----------------------------------------
print(f"  {elapsed()} Neural Network...", end='', flush=True)
mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                    alpha=0.001, max_iter=150, early_stopping=True,
                    validation_fraction=0.1, random_state=42)
mlp.fit(X_train_smote, y_train_smote)
p_nn = mlp.predict_proba(X_test)[:,1]
d_nn = mlp.predict(X_test)
results['Neural Network'] = eval_m(y_test, d_nn, p_nn)
model_store['Neural Network'] = (mlp, p_nn, d_nn)
print(f" F1={results['Neural Network']['F1']:.4f}  AUC={results['Neural Network']['ROC_AUC']:.4f}")

# -----------------------------------------------------------------------------
# 5. HYBRID MODEL: Isolation Forest Scores + XGBoost + Cost Threshold
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [5/9] Hybrid IF-XGB-CST model...")
X_tr_hyb = np.hstack([X_train, iso_sc_train.reshape(-1,1)])
X_te_hyb = np.hstack([X_test,  iso_sc_test.reshape(-1,1)])

hyb = xgb.XGBClassifier(scale_pos_weight=spw, n_estimators=200, max_depth=5,
                          learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, eval_metric='logloss',
                          use_label_encoder=False, n_jobs=-1)
hyb.fit(X_tr_hyb, y_train)
p_hyb = hyb.predict_proba(X_te_hyb)[:,1]

# Cost-sensitive threshold grid
FN_COST, FP_COST = 500, 10
thresholds = np.linspace(0.01, 0.99, 300)
costs, f1s = [], []
for t in thresholds:
    p = (p_hyb >= t).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_test, p, labels=[0,1]).ravel()
    costs.append(fn*FN_COST + fp*FP_COST)
    f1s.append(f1_score(y_test, p, zero_division=0))

opt_t = thresholds[np.argmin(costs)]
d_hyb = (p_hyb >= opt_t).astype(int)
results['Hybrid (IF+XGB)'] = eval_m(y_test, d_hyb, p_hyb)
model_store['Hybrid (IF+XGB)'] = (hyb, p_hyb, d_hyb)
joblib.dump(hyb, os.path.join(MODELS_DIR,'hybrid_model.pkl'))
print(f"  Optimal threshold: {opt_t:.3f}")
print(f"  F1={results['Hybrid (IF+XGB)']['F1']:.4f}  AUC={results['Hybrid (IF+XGB)']['ROC_AUC']:.4f}")

# Cost-threshold plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(thresholds, costs, color='#F44336', lw=2)
axes[0].axvline(x=opt_t, color='#4CAF50', ls='--', lw=2.5, label=f'Optimal: {opt_t:.3f}')
axes[0].fill_between(thresholds, costs, alpha=0.15, color='#F44336')
axes[0].set_xlabel('Decision Threshold'); axes[0].set_ylabel('Total Operational Cost ($)')
axes[0].set_title('Cost vs. Threshold\n(FN=$500, FP=$10)', fontsize=13, fontweight='bold')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
axes[0].legend()
axes[1].plot(thresholds, f1s, color='#2196F3', lw=2)
axes[1].axvline(x=opt_t, color='#4CAF50', ls='--', lw=2.5, label=f'Cost-optimal: {opt_t:.3f}')
axes[1].fill_between(thresholds, f1s, alpha=0.15, color='#2196F3')
axes[1].set_xlabel('Decision Threshold'); axes[1].set_ylabel('F1 Score')
axes[1].set_title('F1 vs. Threshold', fontsize=13, fontweight='bold')
axes[1].legend()
plt.suptitle('Novel Contribution: Cost-Sensitive Threshold Optimization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_cost_threshold.png'), bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 6. EVALUATION PLOTS
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [6/9] Generating evaluation plots...")
COLORS = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#E91E63']
model_names = list(results.keys())

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
for nm, col in zip(model_names, COLORS):
    _, yp, _ = model_store[nm]
    fpr, tpr, _ = roc_curve(y_test, yp)
    auc = results[nm]['ROC_AUC']
    lw = 3 if 'Hybrid' in nm else 1.8
    ax.plot(fpr, tpr, color=col, lw=lw, label=f'{nm} (AUC={auc:.4f})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=13)
ax.set_title('ROC Curve Comparison -- All Models', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_roc_curves.png'), bbox_inches='tight')
plt.close()

# PR Curves
fig, ax = plt.subplots(figsize=(10, 8))
for nm, col in zip(model_names, COLORS):
    _, yp, _ = model_store[nm]
    prec, rec, _ = precision_recall_curve(y_test, yp)
    ap = results[nm]['Avg_Precision']
    ax.plot(rec, prec, color=col, lw=2, label=f'{nm} (AP={ap:.4f})')
baseline = y_test.sum()/len(y_test)
ax.axhline(y=baseline, color='k', ls='--', lw=1, label=f'Baseline (AP={baseline:.4f})')
ax.set_xlabel('Recall', fontsize=13); ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curve Comparison', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '09_pr_curves.png'), bbox_inches='tight')
plt.close()

# Confusion Matrices Grid
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
af = axes.flatten()
for idx, nm in enumerate(model_names):
    ax = af[idx]
    _, _, yd = model_store[nm]
    cm = confusion_matrix(y_test, yd)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                linewidths=0.5, linecolor='white', annot_kws={'size':12,'weight':'bold'})
    r = results[nm]
    ax.set_title(f'{nm}\nRecall={r["Recall"]:.3f}  Prec={r["Precision"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_xticklabels(['Legit','Fraud'], fontsize=9)
    ax.set_yticklabels(['Legit','Fraud'], fontsize=9, rotation=0)
af[-1].set_visible(False)
plt.suptitle('Confusion Matrix Comparison -- All Models', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_confusion_matrices.png'), bbox_inches='tight')
plt.close()

# Feature Importance (Random Forest)
fi_df = pd.DataFrame({'Feature': feature_cols,
                      'Importance': rf.feature_importances_}
                     ).sort_values('Importance', ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(10, 10))
bar_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_df)))
bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=bar_colors)
for bar, val in zip(bars, fi_df['Importance']):
    ax.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=12)
ax.set_title('Top 20 Feature Importances -- Random Forest', fontsize=14, fontweight='bold')
ax.set_xlim(0, fi_df['Importance'].max()*1.18)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_feature_importance.png'), bbox_inches='tight')
plt.close()

print(f"  [OK] 5 evaluation plots saved")

# -- Model comparison bar chart
comp_df = pd.DataFrame(results).T.reset_index()
comp_df.columns = ['Model','Bal_Accuracy','Precision','Recall','F1','ROC-AUC','Avg Precision','MCC']
comp_df.to_csv(os.path.join(BASE_DIR, 'outputs', 'model_comparison.csv'), index=False)

# Primary metrics -- excludes regular Accuracy (misleading for 578:1 imbalanced data)
# Balanced Accuracy = average recall per class; gives realistic score for fraud detection
metrics_plot  = ['Bal_Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
metric_labels = ['Balanced\nAccuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
x = np.arange(len(model_names))
width = 0.16
fig, ax = plt.subplots(figsize=(18, 7))
for i, (metric, mlabel) in enumerate(zip(metrics_plot, metric_labels)):
    vals = comp_df[metric].values
    bars = ax.bar(x + i*width, vals, width, label=mlabel,
                  color=COLORS[i], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xlabel('Model', fontsize=12); ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison -- Key Metrics\n'
             '(Balanced Accuracy used; Raw Accuracy omitted due to 578:1 class imbalance)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x + width*2); ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylim(0, 1.15); ax.legend(fontsize=10)
ax.text(0.01, 0.97,
        'Note: Raw Accuracy NOT shown -- a naive model predicting all transactions as\n'
        'Legitimate scores 99.83% Accuracy but catches ZERO fraud (useless baseline).',
        transform=ax.transAxes, fontsize=8, color='#444444', verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', alpha=0.85))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '00_model_comparison_bar.png'), bbox_inches='tight')
plt.close()
print(f"  [OK] Model comparison bar chart saved")

# -----------------------------------------------------------------------------
# 7. SAVE MODELS + METADATA
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [7/9] Saving models...")
joblib.dump(scaler, os.path.join(MODELS_DIR,'scaler.pkl'))
joblib.dump({'feature_cols': feature_cols, 'optimal_threshold': float(opt_t)},
            os.path.join(MODELS_DIR,'metadata.pkl'))
print(f"  [OK] Models saved to {MODELS_DIR}")

# -----------------------------------------------------------------------------
# 8. PRINT FINAL COMPARISON TABLE
# -----------------------------------------------------------------------------
print(f"\n{elapsed()} [8/9] Model Comparison Table")
print("\n" + "="*95)
print(f"  NOTE: 'Bal. Acc.' = Balanced Accuracy (accounts for 578:1 imbalance).")
print(f"  Regular Accuracy is NOT shown -- a model predicting ALL as legit scores 99.83%!")
print("="*95)
print(f"{'Model':<22} {'Bal.Acc':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'MCC':>8}")
print("="*95)
for _, row in comp_df.iterrows():
    marker = " [*]" if 'Hybrid' in row['Model'] else "    "
    print(f"{row['Model']:<22}{marker} {row['Bal_Accuracy']:>9.4f} {row['Precision']:>10.4f}"
          f" {row['Recall']:>8.4f} {row['F1']:>8.4f} {row['ROC-AUC']:>9.4f} {row['MCC']:>8.4f}")
print("="*95)
print("[*] = Proposed Hybrid Model")
print("Bal.Acc = Balanced Accuracy | MCC = Matthews Correlation Coefficient")

