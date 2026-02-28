# California Housing Price Prediction

## A Complete Machine Learning Regression Project

 ### 1.Project Summary

This project builds and compares multiple machine learning models to predict housing prices in California using structured tabular data.

The goal is to simulate a real-world scenario where a real estate company wants to price properties accurately based on economic and housing features.

Since the target variable (price) is continuous, this is a regression problem.

#### 2.Business Objective

Accurate property pricing is essential in real estate:

Underpricing → revenue loss

Overpricing → slow sales

Inconsistent pricing → reduced trust

This project answers:

Which factors influence house prices the most, and which model predicts prices most accurately?

### 3.Dataset

Source: sklearn.datasets.fetch_california_housing

Observations: ~20,000 housing districts

Features include:

Median Income

House Age

Average Rooms

Population

Latitude & Longitude

Target: Median House Value

### 4.Exploratory Data Analysis (EDA)

EDA was conducted to:

Understand data structure

Detect missing values

Identify skewness

Check feature distributions

Analyze relationships between variables

#### 4.1 Target Distribution

The price distribution was right-skewed:

Most properties fall in lower-to-mid price ranges

Few high-value properties create a long right tail

Skewness confirmed using .skew()

#### 4.2 Log Transformation

To improve regression performance:

df['price_log'] = np.log1p(df['price'])

Why?

Reduces skewness

Reduces outlier influence

Improves normality of residuals

Helps linear models perform better

After transformation, the distribution became more symmetric.

#### 4.3 Feature Relationships
Correlation Matrix

Identified strong predictors (e.g., Median Income)

Checked multicollinearity

Scatter Analysis

A strong positive linear relationship was observed between Median Income and House Price.

However:

Variance increases at higher income levels

Target appears capped at upper values

Indicates potential heteroscedasticity

### 5.Data Splitting

Dataset split into:

80% Training

20% Testing

This ensures model evaluation on unseen data and prevents overfitting.

### 6.Feature Scaling

Applied StandardScaler for linear models:

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

Important:

Fit only on training data

Transform test data using learned parameters

Prevents data leakage

### 7.Models Implemented
1️⃣ Linear Regression

Baseline model.

2️⃣ Ridge Regression (L2 Regularization)

Handles multicollinearity and reduces overfitting.

3️⃣ Lasso Regression (L1 Regularization)

Performs automatic feature selection.

4️⃣ Decision Tree Regressor

Captures non-linear relationships but prone to overfitting.

5️⃣ Random Forest Regressor

Ensemble model that:

Reduces variance

Handles non-linearity

Requires minimal preprocessing

### 8.Evaluation Metrics

Used:

MAE – Average absolute error

RMSE – Penalizes large errors

R² Score – Variance explained

Model selection was based on:

Lowest RMSE

Highest R²

Small gap between train and test performance

Random Forest achieved the strongest overall performance.

### 9.Hyperparameter Tuning

Used GridSearchCV with 5-fold cross-validation to tune:

n_estimators

max_depth

This improved model stability and predictive accuracy.

### 10.Feature Importance

Extracted from the best Random Forest model.

Key insight:

Median Income is the strongest predictor of house prices.

This step bridges technical modeling with business interpretation.

### 11.Final Model Comparison
Model	MAE	RMSE	R²
Linear Regression			
Ridge			
Lasso			
Decision Tree			
Random Forest			

(Random Forest performed best overall.)
