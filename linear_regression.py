import pandas as pd # to extract the .csv file
import numpy as np # to use math functions
import matplotlib.pyplot as plt # to control over plots
import seaborn as sns # to visualize data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)


# read in the dataset, print first 5 lines

df = pd.read_csv("housing.csv")
print(df.head())

# check for missing data

miss = df.isnull().sum()
miss = miss[miss > 0] # remove columns with zero missing values
if miss.empty:
    print("Không có cột nào bị thiếu dữ liệu.")
else:
    print(pd.DataFrame({
        "missing": miss,
        "percent": (miss / len(df) * 100).round(2)
    }).sort_values("missing", ascending=False))

# eradicate rows with missing values

df = df.dropna()

# analyze the distribution of every numeric column

num_cols = df.select_dtypes(include=np.number).columns
print(df[num_cols].describe())
print(df[num_cols].skew().sort_values(ascending=False))

df[num_cols].hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.savefig("dist_before.png")
plt.close()

# feature engineering: ratios carry more information than raw counts
# (built from the raw values, before any skew transform)

df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# reduce the right skew of the heavily skewed columns

skewed = ["total_rooms", "total_bedrooms", "population", "households"]
for col in skewed:
    df[col] = df[col] ** 0.3
print(df[skewed].skew())

df[num_cols].hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.savefig("dist_after.png")
plt.close()

# one-hot encoding for the only categorical column

df = pd.get_dummies(df, columns=["ocean_proximity"])
print(df.head())

# correlation with the target

corr = df.corr()["median_house_value"].sort_values(ascending=False)
print(corr)

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.tight_layout()
plt.savefig("correlation.png")
plt.close()

# split into features / target, then into train / test

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling: fit on the train set only, then transform both

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# train the model

model = LinearRegression().fit(X_train, y_train)
print(pd.Series(model.coef_, index=X.columns).sort_values(ascending=False))

# evaluate on the unseen test set

pred = model.predict(X_test)
print("MAE :", mean_absolute_error(y_test, pred))
print("MSE :", mean_squared_error(y_test, pred))
print("R2  :", r2_score(y_test, pred))
print("MAPE:", mean_absolute_percentage_error(y_test, pred))

# predicted vs actual

plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred, s=5, alpha=0.3)
plt.plot([0, 500000], [0, 500000], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("pred_vs_actual.png")
plt.close()
