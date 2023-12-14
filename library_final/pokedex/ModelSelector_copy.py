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
from sklearn.model_selection import KFold


import warnings
warnings.filterwarnings("ignore")

### 2. Load Datasets. No need for function since we already have RUI's df.
#pokemon_data = pd.read_csv('/Users/luispoli/Documents/BSE/T1/Computing_DS/Practice/CDS_final_pokemon/raw_data/ProcessedData.csv')

#columns_to_use = ['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack','Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary','Victory_Rate', 'Total_Battles', 'Victory_Counts', 'Offensive_Power','Defensive_Power', 'Speed_to_Power_Ratio', 'Bug', 'Dark', 'Dragon','Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass','Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel','Water']
#pokemon_data = pokemon_data[columns_to_use]


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

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def make_pred(self, X_test):
        return self.model.predict(X_test)

    def model_evaluation(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        r_sqr = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse} and R-squared: {r_sqr}")
        print("   ")


class LinearRegressionAnalyzer(Model):
    def __init__(self):
        super().__init__(LinearRegression())  # Create an instance of LinearRegression

    def run_model(self, X_train, y_train, X_test, y_test):
        self.train_model(X_train, y_train)
        y_pred = self.make_pred(X_test)
        self.model_evaluation(y_test, y_pred)

    def k_fold_cross_validation(self, df_X, df_y):
        lin_scoring = {'r_squared': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)}
        cv_results = cross_validate(self.model, df_X, df_y, cv=20, scoring=lin_scoring)

        for metric in lin_scoring:
            print("   ")
            print(f"Cross-Validation {metric} Scores: {cv_results[f'test_{metric}']}")
            print(f"Mean Cross-Validation {metric}: {np.mean(cv_results[f'test_{metric}'])}\n")
    


class LassoModel(Model):
    def __init__(self, alphas=None):
        if alphas is None:
            alphas = [0.1, 0.5, 1.0]
        self.alphas = alphas
        super().__init__(Lasso(alpha=self.alphas[0]))  # Create an instance of Lasso

    def tune_alpha(self, df_X, df_y, n_splits=5):
        kf = KFold(n_splits=n_splits)
        lasso_cv = LassoCV(alphas=self.alphas, cv=kf)
        lasso_cv.fit(df_X, df_y)
        self.model.alpha = lasso_cv.alpha_
        print(f'Optimal alpha after tuning: {self.model.alpha}')

    def run_model_cv(self, df_X, df_y, n_splits=5):
        self.tune_alpha(df_X, df_y, n_splits)
        super().run_model_cv(df_X, df_y, n_splits)


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



# Example usage
#rf_wrapper = RandomForestRegressorWrapper(X_train, y_train, X_test, y_test, columns_to_use)
#rf_wrapper.fit_and_predict()








