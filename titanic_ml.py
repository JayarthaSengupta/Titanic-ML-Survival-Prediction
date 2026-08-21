# titanic_ml.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the training data
data = pd.read_csv("data/train.csv")

# 2. Drop unused columns
data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# 3. Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# 4. Encode categorical columns
# Convert 'Sex' to numeric
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})

# One-hot encode 'Embarked'
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)

# 5. Define features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# 6. Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Logistic Regression
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_val)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr))

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_val, y_pred_lr)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("images/logistic_regression_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 8. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_val, y_pred_rf)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("images/random_forest_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 9. (Optional) Predict on test.csv for submission
test_data = pd.read_csv("data/test.csv")
test_passenger_id = test_data["PassengerId"]
test_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing values
test_data["Age"].fillna(data["Age"].median(), inplace=True)
test_data["Fare"].fillna(data["Fare"].median(), inplace=True)

# Encode same as training data
test_data["Sex"] = test_data["Sex"].map({"male": 1, "female": 0})
test_data = pd.get_dummies(test_data, columns=["Embarked"], drop_first=True)

# Align test columns with training
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Final predictions (using Random Forest)
final_preds = rf.predict(test_data)
submission = pd.DataFrame({"PassengerId": test_passenger_id, "Survived": final_preds})
submission.to_csv("data/submission.csv", index=False)
print("\nSubmission file saved as submission.csv")
