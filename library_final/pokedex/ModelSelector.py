### 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor



import warnings
warnings.filterwarnings("ignore")


### 3. Divide Data into subsets
class TrainTestDivider:
    def __init__(self, df):
        self.df = df

    def train_test(self):
        # Assuming df_X and df_y are attributes you want to access outside the method
        self.df_X = self.df.drop(['Victory_Rate', 'Total_Battles', 'Victory_Counts', 'Type 1', 'Type 2', 'Name'], axis=1)
        self.df_y = self.df['Victory_Rate']

        # 20% split into test data.
        X_train, X_test, y_train, y_test = train_test_split(self.df_X, self.df_y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, self.df_X, self.df_y

class Model:
    def __init__(self, model):
        self.model = model
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def set_data(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def make_pred(self):
        return self.model.predict(self.X_test)

    def model_evaluation(self):
        y_pred = self.make_pred()
        mse = mean_squared_error(self.y_test, y_pred)
        r_sqr = r2_score(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse} and R-squared: {r_sqr}")
        return mse, r_sqr
      
    def simple_result(self):
        self.train_model()
        self.make_pred()
        self.model_evaluation()

    def plot_actual_vs_predicted(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.make_pred(), alpha=0.7)
        sns.regplot(x=self.y_test, y=self.make_pred(), scatter=False, color='red', label='Best Fitted Line')
        plt.title('Actual vs. Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

    def k_fold_cross_validation(self, n_splits=5):
        if self.X_train is not None and self.y_train is not None:
            kf = KFold(n_splits=n_splits)
            lin_scoring = {'r_squared': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)}
            cv_results = cross_validate(self.model, self.X_train, self.y_train, cv=kf, scoring=lin_scoring)

            for metric in lin_scoring:
                print(f"Cross-Validation {metric} Scores: {cv_results[f'test_{metric}']}")
                print(f"Mean Cross-Validation {metric}: {np.mean(cv_results[f'test_{metric}'])}")
            return cv_results


class LinearRegressionAnalyzer(Model):
    def __init__(self):
        super().__init__(LinearRegression())  # Create an instance of LinearRegression

class LassoModel(Model):
    def __init__(self, alphas=None):
        if alphas is None:
            alphas = [0.1, 0.5, 1.0]
        super().__init__(LassoCV(alphas=alphas, cv=5))  # Create an instance of LassoCV

    def tune_alpha(self):
        if self.X_train is not None and self.y_train is not None:
            self.model.fit(self.X_train, self.y_train)
            optimal_alpha = self.model.alpha_
            print(f'Optimal alpha after tuning: {optimal_alpha}')
            return optimal_alpha

    def run_model_cv(self):
        optimal_alpha = self.tune_alpha()

        # Get alphas and their respective MSE
        alphas, mses = [], []
        for alpha in self.model.alphas_:
            self.model.alpha_ = alpha
            mse = np.mean(cross_val_score(self.model, self.X_train, self.y_train, scoring='neg_mean_squared_error', cv=5))
            alphas.append(alpha)
            mses.append(-mse)

        # Plot alphas vs. MSE
        plt.figure(figsize=(8, 6))
        plt.plot(alphas, mses, marker='o')
        plt.xscale('log')  # Log scale for better visualization
        plt.title('Alphas vs. Mean Squared Error (MSE)')
        plt.xlabel('Alpha')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.show()

        return optimal_alpha

class RidgeModel(Model):
    def __init__(self, alphas=None):
        if alphas is None:
            alphas = [0.1, 0.5, 1.0]
        self.alphas = alphas
        super().__init__(Ridge(alpha=self.alphas[0]))  # Create an instance of Ridge

    def tune_alpha(self, n_splits=5):
        kf = KFold(n_splits=n_splits)
        ridge_cv = RidgeCV(alphas=self.alphas, cv=kf)
        ridge_cv.fit(self.X_train, self.y_train)
        optimal_alpha = ridge_cv.alpha_
        print(f'Optimal alpha after tuning: {optimal_alpha}')
        self.model.alpha = optimal_alpha

        # Get alphas and their respective MSE
        alphas, mses = [], []
        for alpha in self.alphas:
            self.model.alpha = alpha
            mse = np.mean(cross_val_score(self.model, self.X_train, self.y_train, scoring='neg_mean_squared_error', cv=5))
            alphas.append(alpha)
            mses.append(-mse)

        # Plot alphas vs. MSE
        plt.figure(figsize=(8, 6))
        plt.plot(alphas, mses, marker='o')
        plt.xscale('log')  # Log scale for better visualization
        plt.title('Alphas vs. Mean Squared Error (MSE)')
        plt.xlabel('Alpha')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.show()

        return optimal_alpha

    def run_model_cv(self, n_splits=5):
        optimal_alpha = self.tune_alpha(n_splits)
        return super().k_fold_cross_validation(n_splits)

class RandomForestModel(Model):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        super().__init__(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))

    def plot_feature_importances(self, feature_names):
        if not hasattr(self.model, 'feature_importances_'):
            print("The underlying model doesn't support feature importances.")
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')  # Rotate labels
        plt.title("Feature Importances")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()




# Assuming you have a dataframe named pokemon_data
# divider = TrainTestDivider(pokemon_data)
# X_train, X_test, y_train, y_test, df_X, df_y = divider.train_test()

# # # Creating a linspace for Lasso alphas
# lasso_alphas = np.linspace(0.001, 1, num=30)

# # Creating a linspace for Ridge alphas
# #ridge_alphas = np.linspace(0.01, 100, num=30)



# #LINEAR REGRESSION
# # # Create an instance of LinearRegressionAnalyzer
# # linear_reg_model = LinearRegressionAnalyzer()

# # # Set your training and testing data
# # linear_reg_model.set_data(X_train, y_train, X_test, y_test)

# # # Train the linear regression model
# #linear_reg_model.simple_result

# # # Make predictions
# # predictions = linear_reg_model.make_pred()

# # # Evaluate the model
# # mse, r_squared = linear_reg_model.model_evaluation()

# # # Optionally, you can perform k-fold cross-validation
# # cv_results = linear_reg_model.k_fold_cross_validation()

# # linear_reg_model.plot_actual_vs_predicted()





# #LASSO
# lasso_instance = LassoModel(alphas=lasso_alphas)

# # Set the data
# lasso_instance.set_data(X_train, y_train, X_test, y_test)


# lasso_instance.simple_result()

# #Run the Lasso model and perform cross-validation
# lasso_instance.run_model_cv()



# # #RIDGE
# # # Create an instance of RidgeModel
# # ridge_model = RidgeModel(alphas=ridge_alphas)  # You can customize the alphas as needed

# # # Set your training and testing data
# # ridge_model.set_data(X_train, y_train, X_test, y_test)

# # # Train the Ridge model
# # ridge_model.train_model()

# # # Make predictions
# # ridge_predictions = ridge_model.make_pred()

# # ridge_model.run_model_cv()



# #RANDOM FOREST
# # Create an instance of RandomForestModel
# #rf_model = RandomForestModel(n_estimators=100, max_depth=None, random_state=42)

# # Set your training and testing data
# #rf_model.set_data(X_train, y_train, X_test, y_test)

# # Train the Random Forest model
# #rf_model.train_model()

# # Make predictions
# #rf_predictions = rf_model.make_pred()

# # Evaluate the model
# #rf_mse, rf_r_squared = rf_model.model_evaluation()

# # Optionally, perform k-fold cross-validation
# #rf_cv_results = rf_model.k_fold_cross_validation()

# # Plot feature importances
# #feature_names = X_train.columns  
# #rf_model.plot_feature_importances(feature_names)
