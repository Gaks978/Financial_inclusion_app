import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")

# Drop ID column and rows with missing values
df.drop(columns=["uniqueid"], inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Remove age outliers
Q1 = df['age_of_respondent'].quantile(0.25)
Q3 = df['age_of_respondent'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['age_of_respondent'] < (Q1 - 1.5 * IQR)) | 
          (df['age_of_respondent'] > (Q3 + 1.5 * IQR)))]

# Encode target
df['bank_account'] = df['bank_account'].map({'Yes': 1, 'No': 0})

# Encode features
cat_columns = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_columns, drop_first=True)

# Split dataset
X = df.drop("bank_account", axis=1)
y = df["bank_account"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a smaller model
model = RandomForestClassifier(
    n_estimators=50,     # Reduced from 100 to 50
    max_depth=10,        # Limits tree depth (simpler model)
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model (this should now be <100MB)
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Lightweight model saved as model.pkl")


