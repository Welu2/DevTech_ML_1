import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("Customer-Churn.csv")
print(data.head())
print(data.isnull().sum())
print(data.describe())
print(data.columns)
print(data['Churn'].value_counts())

sns.countplot(x="Churn", data=data)
plt.show()  # Ensure to display the plot

# Gender-based churn analysis
# sns.countplot(x='gender', hue='Churn', data=data)
# plt.show()  # Uncomment to show the gender-based churn plot

# Tenure and MonthlyCharges histograms based on Churn
features = ['tenure', 'MonthlyCharges']
fig, ax = plt.subplots(1, 2, figsize=(28, 8))
data[data.Churn == 'No'][features].hist(bins=20, color='red', alpha=0.5, ax=ax)
data[data.Churn == 'Yes'][features].hist(bins=20, color='blue', alpha=0.5, ax=ax)
plt.show()

# Data cleaning and preprocessing
data_clean = data.drop('customerID', axis=1)

# Handle deprecated warnings
for i in data_clean.columns:
    if data_clean[i].dtype == 'object':
        data_clean[i] = LabelEncoder().fit_transform(data_clean[i])

print(data_clean.dtypes)
print(data_clean.head())

X = data_clean.drop('Churn', axis=1)
y = data_clean['Churn']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))

# Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
