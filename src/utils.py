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
