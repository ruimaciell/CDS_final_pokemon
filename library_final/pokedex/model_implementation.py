### 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore")




### 4. Models
class LinearRegressionAnalyzer:
    def __init__(self, X_train, y_train, df_X, df_y):
        self.X_train = X_train
        self.y_train = y_train
        self.df_X = df_X
        self.df_y = df_y
        self.lin_model = LinearRegression()

    def train_model(self):
        self.lin_model.fit(self.X_train, self.y_train)

    def make_predictions(self, X_test):
        return self.lin_model.predict(X_test)

    def evaluate_model(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

    def k_fold_cross_validation(self):
        lin_scoring = {'r_squared': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)}
        cv_results = cross_validate(self.lin_model, self.df_X, self.df_y, cv=20, scoring=lin_scoring)

        for metric in lin_scoring:
            print(f"Cross-Validation {metric} Scores: {cv_results[f'test_{metric}']}")
            print(f"Mean Cross-Validation {metric}: {np.mean(cv_results[f'test_{metric}'])}\n")
    
    def plot_actual_vs_predicted(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', label='Best Fitted Line')
        plt.title('Actual vs. Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

    def plot_residual_distribution(self, y_test, y_pred):
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.show()


class LassoOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_alpha = None
        self.min_mse = float('inf')

    def optimize_alpha(self, alphas):
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, tol=1e-2)
            lasso.fit(self.X_train, self.y_train)
            y_pred_lasso = lasso.predict(self.X_test)
            mse_lasso = mean_squared_error(self.y_test, y_pred_lasso)

            if mse_lasso < self.min_mse:
                self.min_mse = mse_lasso
                self.best_alpha = alpha

        print(f'Optimal alpha for Lasso without CV: {self.best_alpha} and its average MSE: {self.min_mse}')

    def optimize_alpha_cv(self, alphas):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        lasso_cv = LassoCV(alphas=alphas, tol=1e-2, cv=20)
        lasso_cv.fit(X_train_scaled, self.y_train)

        best_alpha_lasso_cv = lasso_cv.alpha_
        y_pred_lasso_cv = lasso_cv.predict(X_test_scaled)
        mse_lasso_cv = mean_squared_error(self.y_test, y_pred_lasso_cv)

        print(f'Optimal alpha for Lasso with CV: {best_alpha_lasso_cv} and its MSE: {mse_lasso_cv}')


class RidgeOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_alpha = None
        self.min_mse = float('inf')
        self.ridge_cv = None  # Initialize ridge_cv as an instance variable

    def optimize_alpha(self, alphas):
        for alpha in alphas:
            ridge = Ridge(alpha=alpha, tol=1e-2)
            ridge.fit(self.X_train, self.y_train)
            y_pred_ridge = ridge.predict(self.X_test)
            mse_ridge = mean_squared_error(self.y_test, y_pred_ridge)

            if mse_ridge < self.min_mse:
                self.min_mse = mse_ridge
                self.best_alpha = alpha

        print(f'Optimal alpha for Ridge without CV: {self.best_alpha} and its average MSE: {self.min_mse}')

    def optimize_alpha_cv(self, alphas):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        self.ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
        self.ridge_cv.fit(X_train_scaled, self.y_train)

        best_alpha_ridge_cv = self.ridge_cv.alpha_
        y_pred_ridge_cv = self.ridge_cv.predict(X_test_scaled)
        mse_ridge_cv = mean_squared_error(self.y_test, y_pred_ridge_cv)

        print(f'Optimal alpha for Ridge with CV: {best_alpha_ridge_cv} and its MSE: {mse_ridge_cv}')

    def plot_alphas_mse(self, alphas):
        if self.ridge_cv is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(alphas, np.mean(self.ridge_cv.cv_values_, axis=0), marker='o', linestyle='-')
            plt.xscale('log')  # Use a logarithmic scale for better visualization
            plt.title('MSE for Different Alpha Values in RidgeCV')
            plt.xlabel('Alpha')
            plt.ylabel('Mean Squared Error')
            plt.grid(True)
            plt.show()
        else:
            print("Please run optimize_alpha_cv first to generate RidgeCV results.")


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

class RandomForestRegressorWrapper:
    def __init__(self, X_train, y_train, X_test, y_test, columns_to_use, n_estimators=100, random_state=0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.columns_to_use = columns_to_use
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.y_pred = None
        self.mse = None
        self.feature_importances = None

    def fit_and_predict(self):
        # Fit the regressor with training data
        self.regressor.fit(self.X_train, self.y_train)

        # Predict on the test set
        self.y_pred = self.regressor.predict(self.X_test)

        # Calculate Mean Squared Error
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        print("Mean Squared Error:", self.mse)

        # Calculate R-squared
        r2 = r2_score(self.y_test, self.y_pred)
        print("R-squared:", r2)

        # Calculate Feature Importances
        self.feature_importances = self.regressor.feature_importances_

        # Check if the number of features is consistent
        if len(self.feature_importances) == self.X_train.shape[1]:
            # Get the indices that would sort the feature importances
            indices = np.argsort(self.feature_importances)[::-1]

            # Print and visualize feature importances
            print("Feature Importances:")
            for f, importance in zip(indices, self.feature_importances):
                print(f"{self.columns_to_use[f]}: {importance}")

            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.bar(range(self.X_train.shape[1]), self.feature_importances[indices])
            plt.xticks(range(self.X_train.shape[1]), [self.columns_to_use[i] for i in indices], rotation=45)
            plt.xlabel("Feature Name")
            plt.ylabel("Importance Score")
            plt.show()

            # Choose an index for the tree you want to visualize (e.g., index 0)
            tree_index = 0

            # Plot the selected tree
            plt.figure(figsize=(20, 10))
            plot_tree(self.regressor.estimators_[tree_index], filled=True, feature_names=[f'Feature_{i}' for i in range(self.X_train.shape[1])])
            plt.title(f'Decision Tree {tree_index}')
            plt.show()
        else:
            print("Inconsistent number of features.")

    def get_predictions(self):
        return self.y_pred

    def get_mse(self):
        return self.mse

    def get_feature_importances(self):
        return self.feature_importances
















# # Assuming you have X_train, y_train, df_X, and df_y defined
# lr_analyzer = LinearRegressionAnalyzer(X_train, y_train, df_X, df_y)

# # Train the model
# lr_analyzer.train_model()

# # Make predictions
# y_pred_lr = lr_analyzer.make_predictions(X_test)

# # Evaluate the model
# lr_analyzer.evaluate_model(y_test, y_pred_lr)

# # Perform k-fold cross-validation
# lr_analyzer.k_fold_cross_validation()

# #lr_analyzer.plot_actual_vs_predicted(y_test, y_pred_lr)

# # Plot distribution of residuals
# #lr_analyzer.plot_residual_distribution(y_test, y_pred_lr)


# # Example usage:
# # Assuming you have X_train, y_train, X_test, y_test, and alphas defined
# lasso_optimizer = LassoOptimizer(X_train, y_train, X_test, y_test)

# # Define a range of alpha values to try
# alphas_lasso = np.linspace(0.0001, 1, 100)

# # Optimize alpha
# lasso_optimizer.optimize_alpha(alphas_lasso)

# # Optimize alpha with CV
# lasso_optimizer.optimize_alpha_cv(alphas_lasso)


# ridge_optimizer = RidgeOptimizer(X_train, y_train, X_test, y_test)

# # Define a range of alpha values to try
# alphas_ridge = np.linspace(0.1, 10, 100)

# # Optimize alpha
# ridge_optimizer.optimize_alpha(alphas_ridge)

# # Optimize alpha with CV
# ridge_optimizer.optimize_alpha_cv(alphas_ridge)

# ridge_optimizer.plot_alphas_mse(alphas_ridge)


# # Example usage
# rf_wrapper = RandomForestRegressorWrapper(X_train, y_train, X_test, y_test, columns_to_use)
# rf_wrapper.fit_and_predict()








