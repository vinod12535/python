import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

cancer = load_breast_cancer()
X = cancer.data 
y = cancer.target 
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y
print(df.head())


print("Missing values before handling:\n", df.isnull().sum())
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])
r1 = pd.get_dummies(df, columns=['target'],drop_first=False)
print(r1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.2, random_state=42, stratify=y)

log_reg = LogisticRegression(max_iter=500, solver="lbfgs")
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred,
target_names=cancer.target_names))