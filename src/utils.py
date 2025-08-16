#exploring messages column
#importing spacy to perform the text preprocessing
import pandas as pd
import numpy as np
import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import spacy

nlp = spacy.load("en_core_web_sm")

from sklearn.metrics import precision_score, accuracy_score

def text_processing(text):
    """This function helps to transform the text data to lemmatised and clear tokens"""
    try:
        #lower letters
        text = text.lower()


        # removing punctuation and stop words
        cleaned_text = [token for token in nlp(text) if not token.is_stop and not token.is_punct]

        #stemming
        lemmas = [token.lemma_ for token in cleaned_text]

        return " ".join(lemmas)
    except Exception as e:
        logging.info("Error occured in utils text processing function")
        raise CustomException(e,sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.info("Error Occured in save object function")
        raise CustomException(e,sys)        

def model_evaluation(X_train, y_train, X_test, y_test, models):
    try:
        model_report = {}
        for i, (name, model) in enumerate(models.items()):
            #Train model
            model.fit(X_train, y_train)

            #prediction
            y_pred = model.predict(X_test)

            #accuracy
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            model_report[name] = {
                "accuracy_score" : accuracy,
                "precision_score": precision
            }

        return model_report

    except Exception as e:
        raise CustomException(e,sys)
    

def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Error Occured in loading pickle file")
        raise CustomException(e,sys)
