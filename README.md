# Real Estate Price Prediction using Machine Learning

This project presents a machine learning-based system to predict global real estate prices using a combination of advanced regression techniques, including ensemble and neural network models.

## Overview

Real estate markets are complex, high-dimensional, and often non-linear. To address this, we built and evaluated four types of models:
- Linear Regression (Baseline)
- Random Forest Regressor
- Neural Network Regressor (MLP)
- Stacked Regressor combining Random Forest and Neural Network

The project aimed to identify the most effective model in terms of accuracy and generalizability for global property price prediction.

## Key Features
- Dataset of 147,000 global property listings from Kaggle
- Feature Engineering (e.g., `price_per_m2`)
- Feature Importance using Random Forest
- Extensive preprocessing:
  - Outlier handling (5th/95th percentile capping)
  - One-hot encoding for categorical variables
  - Unit conversion (e.g., "m²" to float)
  - Standard scaling for numerical features

## Technologies and Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Models and Performance

| Model                   | MSE         | MAE       | R² Score |
|------------------------|-------------|-----------|----------|
| Linear Regression       | Extremely High | Extremely High | -3.2e+17 |
| Random Forest           | 1.26e+08    | 5950.38   | 0.9963   |
| Optimized Random Forest | 8.92e+07    | 4898.25   | 0.9974   |
| Neural Network          | 4.76e+09    | 34726.40  | 0.8602   |
| Stacked Model           | 1.46e+08    | 5921.30   | 0.9957   |

Best performing model: Optimized Random Forest

## Visualizations
- Feature importance bar plots
- Prediction vs Actual value plots for each model
- Correlation heatmaps and scatterplots

## Results and Insights
- Optimized Random Forest achieved the highest R² and lowest error
- Stacked Model showed competitive performance
- Linear Regression failed to capture non-linear relationships
- Neural Network required heavy tuning and underperformed

## Future Work
- Explore more advanced ensemble techniques (e.g., Gradient Boosting, XGBoost)
- Expand dataset with time-series and geospatial features
- Improve interpretability with SHAP or LIME

## How to Run the Notebook
1. Clone or download the repository
2. Open the `.ipynb` file in Jupyter Notebook or Google Colab
3. Run cells sequentially after installing dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
