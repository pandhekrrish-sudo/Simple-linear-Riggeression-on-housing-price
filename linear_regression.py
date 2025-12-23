#  linear regression on housing price -2 
import pandas as pd
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler

# Load housing dataset
df = pd.read_csv(r"C:\Users\puroh\Downloads\BostonHousing.csv")

# Feature selection (use .copy() to avoid warning)
X = df[["rm", "lstat"]].copy()
y = df["medv"].copy()

# Handle missing values (safe assignment)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Normalization
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Correlation
print("Correlation rm vs Price:", X_scaled["rm"].corr(y))
print("Correlation lstat vs Price:", X_scaled["lstat"].corr(y))

# Linear Regression (using rm feature)
slope, intercept, r_value, p_value, std_err = linregress(X_scaled["rm"], y)

print("\nSlope:", slope)
print("Intercept:", intercept)
print("R-value:", r_value)
print("P-value:", p_value)
print("Std Error:", std_err)
