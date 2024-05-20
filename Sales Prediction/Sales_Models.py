import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Sales Prediction/advertising.csv')
data.fillna(method='ffill', inplace=True)

data = pd.get_dummies(data, drop_first=True)

X = data.drop('Sales', axis=1)
Y = data['Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scal = StandardScaler()
X_train = scal.fit_transform(X_train)
X_test = scal.transform(X_test)

lRegression = LinearRegression()
lRegression.fit(X_train, Y_train)
lRegression_Y_pred = lRegression.predict(X_test)

lRegression_MSE = mean_squared_error(Y_test, lRegression_Y_pred)
lRegression_r2 = r2_score(Y_test, lRegression_Y_pred)
print(f'Linear Regression - Mean Squared Error: {lRegression_MSE}')
print(f'Linear Regression - R2 Score: {lRegression_r2}')

svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, Y_train)
SVM_Y_pred = svm_model.predict(X_test)

SVM_MSE = mean_squared_error(Y_test, SVM_Y_pred)
SVM_r2 = r2_score(Y_test, SVM_Y_pred)
print(f'SVM - Mean Squared Error: {SVM_MSE}')
print(f'SVM - R2 Score: {SVM_r2}')

ANN_model = Sequential()
ANN_model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
ANN_model.add(Dense(units=32, activation='relu'))
ANN_model.add(Dense(units=32, activation='relu'))
ANN_model.add(Dense(units=16, activation='relu'))
ANN_model.add(Dense(units=1))

ANN_model.compile(optimizer='adam', loss='mean_squared_error')
history = ANN_model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

ANN_Y_pred = ANN_model.predict(X_test).flatten()

ANN_MSE = mean_squared_error(Y_test, ANN_Y_pred)
ANN_r2 = r2_score(Y_test, ANN_Y_pred)
print(f'ANN - Mean Squared Error: {ANN_MSE}')
print(f'ANN - R^2 Score: {ANN_r2}')

plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='TV', y='Sales', hue='Sales', palette='coolwarm', size='Sales', sizes=(20, 200), legend=False)
plt.title('Relationship between TV Advertising and Sales')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Radio', y='Sales', hue='Sales', palette='coolwarm', size='Sales', sizes=(20, 200), legend=False)
plt.title('Relationship between Radio Advertising and Sales')
plt.xlabel('Radio Advertising Budget')
plt.ylabel('Sales')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Newspaper', y='Sales', hue='Sales', palette='coolwarm', size='Sales', sizes=(20, 200), legend=False)
plt.title('Relationship between Newspaper Advertising and Sales')
plt.xlabel('Newspaper Advertising Budget')
plt.ylabel('Sales')
plt.show()

def plot_heatmaps(data):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.histplot(data, x='TV', y='Sales', bins=30, pthresh=.1, cmap="mako", cbar=True)
    plt.title('Heatmap of TV Advertising vs Sales')
    plt.xlabel('TV Advertising Budget')
    plt.ylabel('Sales')

    plt.subplot(1, 3, 2)
    sns.histplot(data, x='Radio', y='Sales', bins=30, pthresh=.1, cmap="mako", cbar=True)
    plt.title('Heatmap of Radio Advertising vs Sales')
    plt.xlabel('Radio Advertising Budget')
    plt.ylabel('Sales')

    plt.subplot(1, 3, 3)
    sns.histplot(data, x='Newspaper', y='Sales', bins=30, pthresh=.1, cmap="mako", cbar=True)
    plt.title('Heatmap of Newspaper Advertising vs Sales')
    plt.xlabel('Newspaper Advertising Budget')
    plt.ylabel('Sales')

    plt.tight_layout()
    plt.show()

plot_heatmaps(data)

def plot_results(y_test, y_pred_lr, y_pred_svm, y_pred_ann):
    plt.figure(figsize=(12, 8))
    
    plt.scatter(y_test, y_pred_lr, alpha=0.6, label='Linear Regression', color='blue')
    plt.scatter(y_test, y_pred_svm, alpha=0.6, label='SVM', color='green')
    plt.scatter(y_test, y_pred_ann, alpha=0.6, label='ANN', color='red')
    
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linewidth=2)
    
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Comparison of Actual vs Predicted Sales for Different Models')
    plt.legend()
    plt.show()

plot_results(Y_test, lRegression_Y_pred, SVM_Y_pred, ANN_Y_pred)

def plot_residuals(y_test, y_pred_lr, y_pred_svm, y_pred_ann):
    residuals_lr = y_test - y_pred_lr
    residuals_svm = y_test - y_pred_svm
    residuals_ann = y_test - y_pred_ann
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(y_test, residuals_lr, alpha=0.6, label='Linear Regression Residuals', color='blue')
    plt.scatter(y_test, residuals_svm, alpha=0.6, label='SVM Residuals', color='green')
    plt.scatter(y_test, residuals_ann, alpha=0.6, label='ANN Residuals', color='red')
    
    plt.axhline(y=0, color='black', linewidth=2)
    
    plt.xlabel('Actual Sales')
    plt.ylabel('Residuals')
    plt.title('Residuals Comparison for Different Models')
    plt.legend()
    plt.show()

plot_residuals(Y_test, lRegression_Y_pred, SVM_Y_pred, ANN_Y_pred)

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ANN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

def plot_predicted_distributions(y_pred_lr, y_pred_svm, y_pred_ann):
    plt.figure(figsize=(12, 8))

    sns.histplot(y_pred_lr, kde=True, color='blue', label='Linear Regression Predictions', alpha=0.6)
    sns.histplot(y_pred_svm, kde=True, color='green', label='SVM Predictions', alpha=0.6)
    sns.histplot(y_pred_ann, kde=True, color='red', label='ANN Predictions', alpha=0.6)

    plt.title('Distribution of Predicted Sales')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

plot_predicted_distributions(lRegression_Y_pred, SVM_Y_pred, ANN_Y_pred)

def plot_residuals_boxplot(residuals_lr, residuals_svm, residuals_ann):
    residuals_df = pd.DataFrame({
        'Linear Regression': residuals_lr,
        'SVM': residuals_svm,
        'ANN': residuals_ann
    })

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=residuals_df, palette='coolwarm')
    plt.title('Boxplot of Residuals for Different Models')
    plt.ylabel('Residuals')
    plt.show()

plot_residuals_boxplot(Y_test - lRegression_Y_pred, Y_test - SVM_Y_pred, Y_test - ANN_Y_pred)