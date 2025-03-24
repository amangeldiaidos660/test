import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

file_path = r".\sales\sales23-1.xlsx"
df = pd.read_excel(file_path)
print("Информация о данных:")
print(df.info())

df.replace('-', np.nan, inplace=True)

numeric_cols = ["Quantity", "Discount", "Price"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

df["Дата"] = pd.to_datetime(df["Дата"], errors='coerce')
df.dropna(subset=["Дата"], inplace=True)

df["Year"] = df["Дата"].dt.year
df["Month"] = df["Дата"].dt.month
df["DayOfWeek"] = df["Дата"].dt.dayofweek

df = df[df["Quantity"] > 0]

plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляция между переменными")
plt.show()

X = df[["Discount", "Price", "Month", "DayOfWeek"]]
y = df["Quantity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nОценка модели:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

coefficients = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nКоэффициенты модели:")
print(coefficients)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Фактические значения")
plt.ylabel("Предсказанные значения")
plt.title("Фактические vs Предсказанные значения (Линейная регрессия)")
plt.show()
