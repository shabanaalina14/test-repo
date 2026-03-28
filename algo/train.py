import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import pickle

# Load dataset
df = pd.read_csv("../data/bigmart_data.csv")

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("sales", axis=1)
y = df["sales"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("XGBoost R2 Score:", r2_score(y_test, y_pred))

# Save model
pickle.dump(model, open("../model/xgb_model.pkl", "wb"))