import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Energy_data.csv")

print("Dataset Preview:")

print(data.head())

print("\nMissing Values:")

print(data.isnull().sum())

data = data.dropna()


X = data[['temperature', 'hour', 'day', 'month']]

y = data['energy_consumption']


X_train, X_test, y_train, y_test = train_test_split(

 X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    
   print(f"\n{name} Performance:")
   
   print("MAE :", mean_absolute_error(y_true, y_pred))
   
   print("MSE :", mean_squared_error(y_true, y_pred))
   
   print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
   
   print("R2  :", r2_score(y_true, y_pred))

evaluate_model("Linear Regression", y_test, y_pred_lr)

evaluate_model("Random Forest", y_test, y_pred_rf)


EMISSION_FACTOR = 0.82  

co2_emission = y_pred_rf * EMISSION_FACTOR


results = pd.DataFrame({
    "Actual Energy (kWh)": y_test.values,
    "Predicted Energy (kWh)": y_pred_rf,
    "Predicted CO2 Emission (kg)": co2_emission
})

print("\nPrediction Results:")

print(results.head())




plt.figure()

plt.plot(results["Actual Energy (kWh)"].values[:50], label="Actual")

plt.plot(results["Predicted Energy (kWh)"].values[:50], label="Predicted")

plt.title("Energy Consumption Prediction")

plt.xlabel("Samples")

plt.ylabel("Energy (kWh)")

plt.legend()

plt.show()


plt.figure()

plt.plot(results["Predicted CO2 Emission (kg)"].values[:50])

plt.title("Predicted CO2 Emission")

plt.xlabel("Samples")

plt.ylabel("CO2 Emission (kg)")

plt.show()


data["Predicted Energy"] = rf.predict(scaler.transform(X))

data["CO2 Emission"] = data["Predicted Energy"] * EMISSION_FACTOR

monthly_emission = data.groupby("month")["CO2 Emission"].mean()

plt.figure()

monthly_emission.plot(kind="line", marker='o')

plt.title("Monthly Average CO2 Emission")

plt.xlabel("Month")

plt.ylabel("CO2 Emission (kg)")

plt.show()

