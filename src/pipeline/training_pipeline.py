import pandas as pd
import sys
import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
logger = logging.getLogger(__name__)
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test,_,_ =  data_transformation.data_transformation_initiated(train_data,test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, y_train,X_test, y_test)