
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


class CustomModel:
    def __init__(self,df_train, feature_columns, target_column, model_type= None):
        self._feature_columns = feature_columns
        self._target_column = target_column
        self.model_type = None  # Public attribute
        self._df_train = df_train

        # Choose between Logistic Regression and Random Forest Classifier
        if model_type is not None:
            if model_type == "Logistic":
                self.model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
            elif model_type == "RandomForest":
                self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
            else:
                raise ValueError("Invalid model_type. Supported values are 'Logistic' and 'RandomForest'.")
        else:
            raise ValueError("You must specify a model_type (e.g., 'Logistic' or 'RandomForest').")

    def train(self):
        if self._df_train is None:
            raise ValueError("Training data is not set.")
        X_train = self._df_train[self._feature_columns]
        y_train = self._df_train[self._target_column]

        # Fit the model to the training data and store it as an instance variable
        self.model.fit(X_train, y_train)

    def prediction(self,data):
        X_data = data[self._feature_columns]
        # Get predicted probabilities using the stored model
        probabilities = self.model.predict_proba(X_data)
        predictions = self.model.predict(X_data)
        return probabilities, predictions
    

# Create an instance of CustomModel with training data and feature/target columns
# custom_model = CustomModel(df_train, feature_columns=["feature1", "feature2"], target_column="diabetes_mellitus", model_type="RandomForest")

# Train the model
# custom_model.train()

# Make predictions on test data
# predicted_probabilities = custom_model.predict(df_test)

#This is how you use it




