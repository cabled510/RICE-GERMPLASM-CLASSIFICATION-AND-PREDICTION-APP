"""
Model Training Pipeline — Ghanaian Rice Germplasm ML Study
=============================================================
Trains and compares 6 algorithms across 3 tasks using Stratified 5-Fold CV:
  Task A — Accession Classification   (18 classes)
  Task B — Phenotypic Trait Regression (4 targets)
  Task C — Treatment Classification    (Control vs Stress)

Algorithms: Random Forest, SVM, KNN, XGBoost, Logistic/Ridge, Neural Network (MLP)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

np.random.seed(42)

# LOAD PREPROCESSED DATA

print("=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

xl = pd.ExcelFile(r'C:\Users\pc\OneDrive\Desktop\RiceML\Preprocessed_Rice_Data.xlsx')
task_a = pd.read_excel(xl, sheet_name='Task_A_Classification')
task_b = pd.read_excel(xl, sheet_name='Task_B_Regression')
task_c = pd.read_excel(xl, sheet_name='Task_C_Treatment')
acc_map = pd.read_excel(xl, sheet_name='Accession_Encoding')

print(f"Task A (Classification): {task_a.shape}")
print(f"Task B (Regression)    : {task_b.shape}")
print(f"Task C (Treatment)     : {task_c.shape}")

results_summary = []   


# TASK A — ACCESSION CLASSIFICATION

print("\n" + "=" * 70)
print("TASK A — ACCESSION CLASSIFICATION (18 classes)")
print("=" * 70)

X_a = task_a.drop(columns=['Target_Accession'])
y_a = task_a['Target_Accession']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_classifier(model, X, y, cv, scale=None, name=""):
    """Runs stratified CV, returns mean/std of accuracy, weighted F1, macro F1, kappa"""
    accs, f1w, f1m, kappas = [], [], [], []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if scale == 'std':
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        elif scale == 'mm':
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, pred))
        f1w.append(f1_score(y_test, pred, average='weighted', zero_division=0))
        f1m.append(f1_score(y_test, pred, average='macro', zero_division=0))
        kappas.append(cohen_kappa_score(y_test, pred))

    return {
        'Model': name,
        'Accuracy': f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        'Weighted_F1': f"{np.mean(f1w):.3f} ± {np.std(f1w):.3f}",
        'Macro_F1': f"{np.mean(f1m):.3f} ± {np.std(f1m):.3f}",
        'Kappa': f"{np.mean(kappas):.3f} ± {np.std(kappas):.3f}",
        '_acc_mean': np.mean(accs), '_f1w_mean': np.mean(f1w)
    }

models_a = [
    (RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42), None, "Random Forest"),
    (SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42), 'std', "SVM (RBF)"),
    (KNeighborsClassifier(n_neighbors=5, metric='euclidean'), 'std', "KNN"),
    (XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', random_state=42), None, "XGBoost"),
    (LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42), 'std', "Logistic Regression"),
    (MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', alpha=0.001, max_iter=500, early_stopping=True, random_state=42), 'mm', "Neural Network (MLP)"),
]

results_a = []
for model, scale, name in models_a:
    print(f"  Training {name}...")
    res = evaluate_classifier(model, X_a, y_a, skf, scale, name)
    results_a.append(res)

df_results_a = pd.DataFrame(results_a)
print("\n--- TASK A RESULTS ---")
print(df_results_a[['Model','Accuracy','Weighted_F1','Macro_F1','Kappa']].to_string(index=False))

best_a = max(results_a, key=lambda r: r['_f1w_mean'])
print(f"\nBest model for Task A: {best_a['Model']} (Weighted F1 = {best_a['_f1w_mean']:.3f})")


# TASK B — PHENOTYPIC TRAIT REGRESSION

print("\n" + "=" * 70)
print("TASK B — PHENOTYPIC TRAIT REGRESSION")
print("=" * 70)

targets_b = ['Target_Leaf', 'Target_RootNum', 'Target_RootLen', 'Target_FinalHeight']
feature_cols_b = [c for c in task_b.columns if c not in targets_b]
X_b = task_b[feature_cols_b]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_regressor(model, X, y, cv, scale=None, name=""):
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if scale == 'std':
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        elif scale == 'mm':
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, pred)))
        r2s.append(r2_score(y_test, pred))

    return {
        'Model': name,
        'MAE': f"{np.mean(maes):.3f} ± {np.std(maes):.3f}",
        'RMSE': f"{np.mean(rmses):.3f} ± {np.std(rmses):.3f}",
        'R2': f"{np.mean(r2s):.3f} ± {np.std(r2s):.3f}",
        '_r2_mean': np.mean(r2s)
    }

models_b_template = [
    (lambda: RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42), None, "Random Forest"),
    (lambda: SVR(kernel='rbf', C=10, gamma='scale'), 'std', "SVR"),
    (lambda: XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42), None, "XGBoost"),
    (lambda: Ridge(alpha=1.0, random_state=42), 'std', "Ridge Regression"),
    (lambda: MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', alpha=0.001, max_iter=500, early_stopping=True, random_state=42), 'mm', "Neural Network (MLP)"),
]

all_task_b_results = {}
for target in targets_b:
    print(f"\n  -- Target: {target} --")
    y_b = task_b[target]
    results_b = []
    for model_fn, scale, name in models_b_template:
        model = model_fn()
        res = evaluate_regressor(model, X_b, y_b, kf, scale, name)
        results_b.append(res)
        print(f"     {name:<22} MAE={res['MAE']:<16} R²={res['R2']}")
    all_task_b_results[target] = pd.DataFrame(results_b)

print("\n--- TASK B SUMMARY (best R² per target) ---")
for target, df_r in all_task_b_results.items():
    best = df_r.loc[df_r['_r2_mean'].idxmax()]
    print(f"  {target:<20} Best: {best['Model']:<22} R² = {best['R2']}")



# TASK C — TREATMENT CLASSIFICATION

print("\n" + "=" * 70)
print("TASK C — TREATMENT CLASSIFICATION (Control vs Stress)")
print("=" * 70)

X_c = task_c.drop(columns=['Target_Treatment'])
y_c = task_c['Target_Treatment']

skf_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_c = [
    (RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42), None, "Random Forest"),
    (SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42), 'std', "SVM (RBF)"),
    (KNeighborsClassifier(n_neighbors=5, metric='euclidean'), 'std', "KNN"),
    (XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42), None, "XGBoost"),
    (LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42), 'std', "Logistic Regression"),
    (MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', alpha=0.001, max_iter=500, early_stopping=True, random_state=42), 'mm', "Neural Network (MLP)"),
]

results_c = []
for model, scale, name in models_c:
    print(f"  Training {name}...")
    res = evaluate_classifier(model, X_c, y_c, skf_c, scale, name)
    results_c.append(res)

df_results_c = pd.DataFrame(results_c)
print("\n--- TASK C RESULTS ---")
print(df_results_c[['Model','Accuracy','Weighted_F1','Macro_F1','Kappa']].to_string(index=False))

best_c = max(results_c, key=lambda r: r['_f1w_mean'])
print(f"\nBest model for Task C: {best_c['Model']} (Weighted F1 = {best_c['_f1w_mean']:.3f})")


# FEATURE IMPORTANCE (Random Forest)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE — RANDOM FOREST (Task A)")
print("=" * 70)

rf_a = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_a.fit(X_a, y_a)
importances_a = pd.Series(rf_a.feature_importances_, index=X_a.columns).sort_values(ascending=False)
print(importances_a.head(10).to_string())

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE — RANDOM FOREST (Task C)")
print("=" * 70)
rf_c = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_c.fit(X_c, y_c)
importances_c = pd.Series(rf_c.feature_importances_, index=X_c.columns).sort_values(ascending=False)
print(importances_c.head(10).to_string())


# FEATURE IMPORTANCE (XGBoost)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE — XGBOOST (Task A)")
print("=" * 70)

xgb_a = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', random_state=42)
xgb_a.fit(X_a, y_a)
importances_xgb_a = pd.Series(xgb_a.feature_importances_, index=X_a.columns).sort_values(ascending=False)
print(importances_xgb_a.head(10).to_string())

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE — XGBOOST (Task C)")
print("=" * 70)

xgb_c_full = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42)
xgb_c_full.fit(X_c, y_c)
importances_xgb_c = pd.Series(xgb_c_full.feature_importances_, index=X_c.columns).sort_values(ascending=False)
print(importances_xgb_c.head(10).to_string())



# SAVE ALL RESULTS TO EXCEL

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)
output_path = r'C:\Users\pc\OneDrive\Desktop\RiceML\Model_Training_Results.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_results_a[['Model','Accuracy','Weighted_F1','Macro_F1','Kappa']].to_excel(writer, sheet_name='TaskA_Classification', index=False)
    for target, df_r in all_task_b_results.items():
        sheet_name = f'TaskB_{target.replace("Target_","")}'[:31]
        df_r[['Model','MAE','RMSE','R2']].to_excel(writer, sheet_name=sheet_name, index=False)
    df_results_c[['Model','Accuracy','Weighted_F1','Macro_F1','Kappa']].to_excel(writer, sheet_name='TaskC_Treatment', index=False)
    importances_a.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).to_excel(writer, sheet_name='RF_FeatImport_TaskA', index=False)
    importances_c.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).to_excel(writer, sheet_name='RF_FeatImport_TaskC', index=False)
    importances_xgb_a.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).to_excel(writer, sheet_name='XGB_FeatImport_TaskA', index=False)
    importances_xgb_c.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).to_excel(writer, sheet_name='XGB_FeatImport_TaskC', index=False)

print(f"Saved all results to: {output_path}")
print("\nDONE.")
