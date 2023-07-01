import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_tainer import ModelTrainer
from src.components.model_tainer import ModelTrainerConfig
import pandas as pd 
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path : str= os.path.join('artifacts',"train.csv")
    test_data_path : str= os.path.join('artifacts',"test.csv")
    raw_data_path : str= os.path.join('artifacts',"data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion")
        try:
            df = pd.read_csv('data/Air_Quality.csv')
            logging.info("Read Dataset")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test set split initiated")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of data completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
    
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_trans = DataTransformation()
    X_train_transformed, y_train, X_test_transformed, y_test, _ = data_trans.initiate_data_transform(train_data, test_data)

    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train_transformed, y_train, X_test_transformed, y_test))

    
