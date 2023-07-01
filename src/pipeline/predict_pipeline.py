import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        country: str,
        state: str,
        city: str,
        station: str,
        pollutant_min: int,
        pollutant_max: int):

        self.country = country

        self.state = state

        self.city = city

        self.station = station

        self.pollutant_min = pollutant_min

        self.pollutant_max = pollutant_max

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "country": [self.country],
                "state": [self.state],
                "city": [self.city],
                "station": [self.station],
                "pollutant_min": [self.pollutant_min],
                "pollutant_max": [self.pollutant_max],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)