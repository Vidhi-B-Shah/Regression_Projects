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

    def get_data_transf_obj(self):
        '''
        Function to perform Data Transformation
        '''
        try:
            num_col = ['pollutant_min','pollutant_max']
            cat_col = ['country','state','city','station','pollutant_id']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder())
                ]
            )

            logging.info("Numerical columns scaling and categorical columns encoding done")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_col),
                    ("cat_pipeline", cat_pipeline, cat_col)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining processing object ")
            preprocessing_obj = self.get_data_transf_obj()
            target_col = 'pollutant_avg'
            num_col = ['pollutant_min','pollutant_max']

            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("Applying preprocessing on train and test dataframes ")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Reshape target feature arrays to match the input feature arrays
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            # Convert the arrays to pandas DataFrame
            input_feature_train_df = pd.DataFrame(input_feature_train_arr)
            target_feature_train_df = pd.DataFrame(target_feature_train_arr)
            input_feature_test_df = pd.DataFrame(input_feature_test_arr)
            target_feature_test_df = pd.DataFrame(target_feature_test_arr)

            # Perform the concatenation
            train_df = pd.concat([input_feature_train_df, target_feature_train_df], axis=1)
            test_df = pd.concat([input_feature_test_df, target_feature_test_df], axis=1)

            # Convert the DataFrame back to numpy array, if needed
            train_arr = train_df.values
            test_arr = test_df.values

            
        

            # train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            # test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)


            # train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
            # test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))

            logging.info("Saved preprocessing object ")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        except Exception as e:
            raise CustomException(e, sys)


# print('input_feature_test_arr',input_feature_test_arr.shape)
            # print('input_feature_train_arr',input_feature_train_arr.shape)
            # print('target_feature_train_arr : ',target_feature_train_arr.shape)
            # print('target_feature_test_arr : ',target_feature_test_arr.shape)
            
            # print(np.isnan(input_feature_train_arr).any())
            # print(np.isnan(target_feature_train_arr).any())
            # print(np.isnan(input_feature_test_arr).any())
            # print(np.isnan(target_feature_test_arr).any())