import pickle
import pandas as pd

# Load model
model = pickle.load(open("../model/xgb_model.pkl", "rb"))

# Example input
data = {
    "item_weight": [10],
    "item_mrp": [180],
    "outlet_size": ["Medium"],
    "outlet_location": ["Tier2"],
    "item_type": ["Food"]
}

df = pd.DataFrame(data)

# Same preprocessing
df = pd.get_dummies(df)

# Align columns with training data
model_columns = model.get_booster().feature_names
df = df.reindex(columns=model_columns, fill_value=0)

# Predict
prediction = model.predict(df)

print("Predicted Sales:", prediction[0])