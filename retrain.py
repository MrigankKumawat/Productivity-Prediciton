import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import joblib

df = pd.read_csv("final_productivity_dataset.csv")
df.drop(columns="productivity_label", inplace=True)
df["productivity_score"] = df["productivity_score"].round(2)

X = df.drop("productivity_score", axis=1)
y = df["productivity_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_attribs = X_train.columns.tolist()
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs)])

X_train_prepared = full_pipeline.fit_transform(X_train)

model = GradientBoostingRegressor()
model.fit(X_train_prepared, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(full_pipeline, "pipeline.pkl")
print("Done — model.pkl and pipeline.pkl saved.")