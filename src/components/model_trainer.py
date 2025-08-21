import os
import sys
from dataclasses import dataclass
from src.logger import logging
logger = logging.getLogger(__name__)
from src.exception import CustomException
from src.utils import model_evaluation
from src.utils import save_object
from src.utils import read_yaml
import json
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

params = read_yaml("params.yaml")

class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.metrics_path = os.path.join('artifacts', 'metrics.json')

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        logger.info("Model training initiated")
        try:
            svc = SVC(kernel = params['models']['svc']['kernel'],gamma = params['models']['svc']['gamma'])
            knc = KNeighborsClassifier()
            mnb = MultinomialNB()
            dtc = DecisionTreeClassifier(max_depth = params['models']['dtc']['max_depth'])
            lrc = LogisticRegression(solver = params['models']['lrc']['solver'], penalty = params['models']['lrc']['penalty'])
            rfc = RandomForestClassifier(n_estimators = params['models']['rfc']['n_estimators'], random_state = params['models']['rfc']['random_state'])
            abc = AdaBoostClassifier(n_estimators = params['models']['abc']['n_estimators'], random_state = params['models']['abc']['random_state'])
            bc = BaggingClassifier(n_estimators = params['models']['bc']['n_estimators'], random_state = params['models']['bc']['random_state'])
            etc = ExtraTreesClassifier(n_estimators = params['models']['etc']['n_estimators'], random_state = params['models']['etc']['random_state'])
            gbt = GradientBoostingClassifier(n_estimators = params['models']['gbt']['n_estimators'], random_state = params['models']['gbt']['random_state'])
            xgb = XGBClassifier(n_estimators = params['models']['xgb']['n_estimators'], random_state = params['models']['xgb']['random_state'])

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
            logger.info(f"model training completed : {model_report}")

            best_model_name = max(model_report, key = lambda name : model_report[name]['accuracy_score'])
            best_model_score = model_report[best_model_name]['accuracy_score']

            best_model = models[best_model_name]

            print(f"The best model name {best_model_name} and the accuracy is {best_model_score}")
            print("\n=============")
            logger.info(f"The best model name {best_model_name} and the accuracy is {best_model_score}")
            
            metrics_dict = {
                "Best Model Name'": best_model_name,
                "Best Model Score": best_model_score
            }
            os.makedirs('artifacts', exist_ok=True)
            with open(self.metrics_path,'w') as f:
                json.dump(metrics_dict, f)

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model

            )

        except Exception as e:
            raise CustomException(e,sys)



