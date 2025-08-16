import os 
import sys 
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj

class PredictPipeline:
    """"This is the prediction pipeline class"""
    try:
        def __init__(self):
            pass

        def predict(self,features):
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_processed = preprocessor.transform(features)

            pred = model.predict(data_processed)

            return pred
    except Exception as e:
        logging.info("Error occured at prediction pipeline")
        raise CustomException(e, sys)
    
class CustomData:
    def __init__(self, message: str):
        self.message = message

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'message' : [self.message]
            }

            df = pd.DataFrame(custom_data_input_dict)

            return df
        except Exception as e:
            logging.info("Error occured in creating dataframe")
            raise CustomException(e,sys)
        
    