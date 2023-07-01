import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
            
    def initiate_data_transform(self, train_path, test_path):
        try:
            
            numeric_features = ['pollutant_min', 'pollutant_max']
            categorical_features = ['country', 'state', 'city', 'station']

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
            
            data_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)])
            
            
            
            
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining processing object ")
            
            
            
            

            # separate input features and target variable
            
            X_train = train_df.drop(columns=['pollutant_avg'])
            y_train = train_df['pollutant_avg']
            X_test = test_df.drop(columns=['pollutant_avg'])
            y_test = test_df['pollutant_avg']
            
            y_test.fillna(y_test.mean(), inplace=True)
            y_train.fillna(y_train.mean(), inplace=True)

            # fit on training set and transform both training and test set
            X_train_transformed = data_pipeline.fit_transform(X_train)
            X_test_transformed = data_pipeline.transform(X_test)
            
            

            logging.info("Applying preprocessing on train and test dataframes ")
            
            # if y_test.isnull().any():
            #     raise CustomException("target var contains missing values", sys)
            
            logging.info("Saved preprocessing object ")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessor
            )

            return (
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test,
                self.data_transformation_config.preprocessor_obj_file
            )
        except Exception as e:
            raise CustomException(e, sys)



