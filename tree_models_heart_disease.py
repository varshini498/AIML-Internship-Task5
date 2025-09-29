import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

OUTPUT_FOLDER = "output_plots"

print("--- 1. Data Loading and Preprocessing ---")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")
else:
    print(f"Output folder already exists: {OUTPUT_FOLDER}")

try:
    data = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please ensure the dataset file is in the correct directory.")
    exit()

X = data.drop('target', axis=1)
y = data['target']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Dataset loaded. Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("-" * 50)


print("--- 2. Decision Tree Training and Overfitting Analysis ---")

dt_clf_default = DecisionTreeClassifier(random_state=42)
dt_clf_default.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt_clf_default.predict(X_train))
dt_test_acc_default = accuracy_score(y_test, dt_clf_default.predict(X_test))

print(f"Default DT Training Accuracy: {dt_train_acc:.4f}")
print(f"Default DT Test Accuracy: {dt_test_acc_default:.4f}")
print("Observation: High gap between train and test suggests **overfitting**.")

dt_clf_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_clf_pruned.fit(X_train, y_train)
dt_test_acc_pruned = accuracy_score(y_test, dt_clf_pruned.predict(X_test))
print(f"Pruned DT (Max Depth 4) Test Accuracy: {dt_test_acc_pruned:.4f}")

plt.figure(figsize=(25, 12))
plot_tree(dt_clf_pruned,
          feature_names=X.columns.tolist(),
          class_names=['No Disease', 'Disease'],
          filled=True,
          max_depth=3,
          impurity=False,
          rounded=True)
plt.title("Decision Tree Visualization (Max Depth 3)")
plt.savefig(os.path.join(OUTPUT_FOLDER, '1_decision_tree_pruned.png'))
plt.close()
print(f"Saved Decision Tree plot to: {OUTPUT_FOLDER}/1_decision_tree_pruned.png")


print("-" * 50)


print("--- 3. Random Forest Training and Comparison ---")

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_clf.fit(X_train, y_train)

rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test))
print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")

print("\n--- Model Comparison (Test Accuracy) ---")
print(f"Pruned Decision Tree: {dt_test_acc_pruned:.4f}")
print(f"Random Forest:        {rf_test_acc:.4f}")

if rf_test_acc > dt_test_acc_pruned:
    print("Conclusion: Random Forest (Ensemble Learning) generally performs better.")
else:
    print("Conclusion: The pruned Decision Tree performs comparably to or better than the Random Forest on this split.")
print("-" * 50)


print("--- 4. Interpreting Feature Importances ---")

importances = rf_clf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Top 10 Feature Importances:")
print(feature_importance_df.head(10).to_string(index=False))

plt.figure(figsize=(14, 7))
top_features = feature_importance_df.head(10)
plt.bar(top_features['Feature'], top_features['Importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Feature Importances from Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '2_feature_importances.png'))
plt.close()
print(f"Saved Feature Importances plot to: {OUTPUT_FOLDER}/2_feature_importances.png")

print("-" * 50)


print("--- 5. Evaluating Models using 5-Fold Cross-Validation ---")
N_SPLITS = 5

dt_cv_scores = cross_val_score(dt_clf_pruned, X, y, cv=N_SPLITS, scoring='accuracy')
print(f"DT CV Scores: {dt_cv_scores}")
print(f"Mean DT CV Accuracy: {dt_cv_scores.mean():.4f} (Std Dev: {dt_cv_scores.std():.4f})")

rf_cv_scores = cross_val_score(rf_clf, X, y, cv=N_SPLITS, scoring='accuracy')
print(f"\nRF CV Scores: {rf_cv_scores}")
print(f"Mean RF CV Accuracy: {rf_cv_scores.mean():.4f} (Std Dev: {rf_cv_scores.std():.4f})")

final_conclusion = "Random Forest" if rf_cv_scores.mean() > dt_cv_scores.mean() else "Pruned Decision Tree"
print(f"\nFINAL CONCLUSION: The **{final_conclusion}** is the most robust model based on cross-validation.")