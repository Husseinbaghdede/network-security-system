from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logging import logger

import sys,os 

from networksecurity.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data
from networksecurity.utils.main_utils.utils import evaluate_models


from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def train_model(self,X_train,y_train,X_test,y_test):
        models = {
            "Logistic Regression": LogisticRegression(verbose=1),
            "K-Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1)      
        }
        
        params= {
            "Logistic Regression": {},
            "K-Neighbors": {
                'n_neighbors': [3, 5, 7, 9, 11]
            },
            "Decision Tree": {
                'criterion': ['gini', 'entropy','log_loss']
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "AdaBoost": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }
        
        model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
        
        # get best model score from dict
        best_model_score = max(sorted(model_report.values()))
        
        # get best model name from dict 
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)
        
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        ## track the mlflow
        
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        
        network_model =NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
        
        ## model trainer artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logger.info(f"Model trainer artifact: {model_trainer_artifact}")
        
        
        return model_trainer_artifact
        
        
    def initiate_model_trainer(self)-> ModelTrainerArtifact:
        try:
            
            train_array = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            return self.train_model(x_train,y_train,x_test,y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)