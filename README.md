# Kaggle-Instacart-Competition

# This project is used to compete in the Instacart Kaggle.com competition:
### https://www.kaggle.com/c/instacart-market-basket-analysis/data
# Objective:
### Analyze customer orders over time to predict which previously purchased products will be in a user’s next order.
### The results are judged by the mean F1 score.

# Code Summary:
### The code uses XGBoost in order to predict which previously purchased products will be part of the user's next order.
### There is a substantial amount of data pre-processing and feature creation.
### There is also a module for grid search cross validation, which is run through an optimization routine, to obtain the best mean F1 score.
