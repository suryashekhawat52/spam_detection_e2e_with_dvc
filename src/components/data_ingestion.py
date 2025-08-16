import pandas as pd
import sys
import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
#Initialization of data ingestion process

#creating a data config class that will store the paths
@dataclass 
class DataIngestionConfig:
    """Initializes the data configuration """
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')

class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Process started")
        try:
            df = pd.read_csv("notebooks/spamhamdata.csv", sep = '\t',header = None, names = ['label', 'message'])
            logging.info("Data loaded successfully")

            df = df.drop_duplicates()
            logging.info("Duplicates dropped")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            logging.info("Train Test Split started")

            train_data, test_data = train_test_split(df, test_size=0.2, random_state= 2)

            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion Completed")

            return (self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info("Error Occured in data ingestion process")
            raise CustomException(e,sys)
        
        


