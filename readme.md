The script executes a complete machine learning workflow to classify heart disease using tree-based models.

1. Data Preparation
Loads the Heart Disease Dataset (heart.csv).

Separates features (X) from the target variable (y).

Applies one-hot encoding to prepare categorical features.

Splits the data into training and testing sets.

2. Decision Tree Analysis
Trains an initial Decision Tree Classifier and notes signs of overfitting (high training accuracy vs. lower test accuracy).

Controls overfitting by retraining a Pruned Decision Tree (setting max_depth=4).

Generates a visualization (1_decision_tree_pruned.png) of the tree structure.

3. Random Forest and Comparison
Trains a Random Forest Classifier (an ensemble model).

Compares the Random Forest's test accuracy to the pruned Decision Tree's accuracy.

4. Interpretation and Visualization
Calculates Feature Importances from the Random Forest model.

Identifies the most influential features for prediction.

Generates a visualization (2_feature_importances.png) to show the top 10 features.

5. Robust Evaluation
Performs 5-Fold Cross-Validation on both the pruned Decision Tree and the Random Forest.

The final output determines the most robust model based on the cross-validation mean accuracy.