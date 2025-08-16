import os
import sys
from dataclasses import dataclass
from src. logger import logging
from src.exception import CustomException
from src.utils import model_evaluation
from src.utils import save_object

from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        logging.info("Model training initiated")
        try:
            svc = SVC(kernel = 'sigmoid',gamma = 1.0)
            knc = KNeighborsClassifier()
            mnb = MultinomialNB()
            dtc = DecisionTreeClassifier(max_depth = 5)
            lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1')
            rfc = RandomForestClassifier(n_estimators = 50, random_state = 2)
            abc = AdaBoostClassifier(n_estimators = 50, random_state = 2)
            bc = BaggingClassifier(n_estimators = 50, random_state = 2)
            etc = ExtraTreesClassifier(n_estimators = 50, random_state = 2)
            gbt = GradientBoostingClassifier(n_estimators = 50, random_state = 2)
            xgb = XGBClassifier(n_estimators = 50, random_state = 2)

            models = {

                'SVC' : svc,
                'KNN' : knc,
                'NB' : mnb,
                'DT' : dtc,
                'LR' : lrc,
                'RF' : rfc,
                'Adaboost' : abc,
                'Bgc' : bc,
                'ETC' : etc,
                'GBT' : gbt,
                'XGB' : xgb
                }
            
            model_report: dict = model_evaluation(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n======================")
            logging.info(f"model training completed : {model_report}")

            best_model_name = max(model_report, key = lambda name : model_report[name]['accuracy_score'])
            best_model_score = model_report[best_model_name]['accuracy_score']

            best_model = models[best_model_name]

            print(f"The best model name {best_model_name} and the accuracy is {best_model_score}")
            print("\n=============")
            logging.info(f"The best model name {best_model_name} and the accuracy is {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model

            )

        except Exception as e:
            raise CustomException(e,sys)



