import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.logger import logging
logger = logging.getLogger(__name__)
from src.exception import CustomException
from src.utils import text_processing
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    label_encoder_file_path = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        logger.info("Data Transformation Initiated")
        preprocessor = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(max_features = 100))
        ])
        return preprocessor

    def data_transformation_initiated(self, train_data_path, test_data_path):    
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logger.info("Data read successfully")
            logger.info(f"{train_df.head()}")

            # Label encoding the Target variable
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(train_df['label'])
            y_test = encoder.transform(test_df['label'])
            # Ensure integer type
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            logger.info("Label encoding successful")

            X_train = train_df['message'].apply(text_processing)
            X_test = test_df['message'].apply(text_processing)
            X_train.to_csv(os.path.join('artifacts','X_train'), index = False, header = True)
            X_test.to_csv(os.path.join('artifacts','X_test'), index = False, header =True)
            logger.info("Text preprocessing successful")

            logger.info("Tfidf started")
            preprocessor_obj = self.get_data_transformation_object()
            X_train = preprocessor_obj.fit_transform(X_train).toarray()
            X_test = preprocessor_obj.transform(X_test).toarray()
            logger.info("Tfidf process completed")
    

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessor_obj
            )
            save_object(
                file_path= self.data_transformation_config.label_encoder_file_path,
                obj=encoder
            )
            logger.info("preprocssor pickle file saved")

            return (
                X_train, y_train,
                X_test, y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_file_path
            )
            
        except Exception as e:
            logger.info("Error occured at data transformation")
            raise CustomException(e,sys)


