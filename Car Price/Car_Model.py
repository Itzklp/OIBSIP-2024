import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Car Price/car data.csv')

print(data.head())

for column in data.select_dtypes(include=[np.number]).columns:
    data[column].fillna(data[column].median(), inplace=True)
for column in data.select_dtypes(include=[object]).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

labelencoder = LabelEncoder()
data['Fuel_Type'] = labelencoder.fit_transform(data['Fuel_Type'])
data['Selling_type'] = labelencoder.fit_transform(data['Selling_type'])
data['Transmission'] = labelencoder.fit_transform(data['Transmission'])

X = data[['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
y = data['Selling_Price']

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Scaled, y, test_size=0.2, random_state=42)

lRegression = LinearRegression()
lRegression.fit(X_Train, Y_Train)

Y_Predict = lRegression.predict(X_Test)

MSE = mean_squared_error(Y_Test, Y_Predict)
r2 = r2_score(Y_Test, Y_Predict)

print("Mean Squared Error:", MSE)
print("R-squared Score:", r2)
print("Coefficients:", lRegression.coef_)
print("Intercept:", lRegression.intercept_)

data_predictions = pd.DataFrame({'Actual': Y_Test, 'Predicted': Y_Predict})
print(data_predictions.head())

plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(data)
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

plt.figure(figsize=(12, 8))
plot = sns.countplot(x='Car_Name', data=data)
plt.xticks(rotation=90)
for p in plot.patches:
    plot.annotate(p.get_height(), 
                  (p.get_x() + p.get_width() / 2.0, 
                   p.get_height()), 
                  ha='center', 
                  va='center', 
                  xytext=(0, 5),
                  textcoords='offset points')

plt.title("Count of Cars Based on Car Name")
plt.xlabel("Car Name")
plt.ylabel("Count of Cars")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(Y_Test, Y_Predict)
plt.plot([min(Y_Test), max(Y_Test)], [min(Y_Test), max(Y_Test)], color='red')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Predicted vs Actual Selling Price')
plt.show()

plt.figure(figsize=(10, 6))
residuals = Y_Test - Y_Predict
plt.scatter(Y_Predict, residuals)
plt.hlines(y=0, xmin=min(Y_Predict), xmax=max(Y_Predict), colors='red')
plt.xlabel('Predicted Selling Price')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(x='Car_Name', y='Present_Price', data=data)
plt.xticks(rotation=90)
plt.title('Boxplot of Present Price by Car Name')
plt.xlabel('Car Name')
plt.ylabel('Present Price')
plt.show()