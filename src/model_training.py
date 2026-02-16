import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,processed_data_path = "artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir,exist_ok=True)

        logger.info("Model Training Initialization...")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path , "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path , "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path , "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path , "y_test.pkl"))

            logger.info("Data loaded for Model")
        except Exception as e:
            logger.error(f"Error while loading data for model {e}")
            raise CustomException("Failed to load data for model..")
        
    def  train_model(self):
        try:
            self.model = GradientBoostingClassifier(n_estimators=100 , learning_rate=0.1 , max_depth=3 , random_state=42)
            self.model.fit(self.X_train,self.y_train)

            joblib.dump(self.model , os.path.join(self.model_dir , "model.pkl"))

            logger.info("Model trained and saved sucesfully...")

        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model...")
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)

            accuracy =  accuracy_score(self.y_test , y_pred)
            precision = precision_score(self.y_test , y_pred , average="weighted")
            recall = recall_score(self.y_test , y_pred,  average="weighted")
            f1 = f1_score(self.y_test , y_pred,  average="weighted")


            logger.info(f"Accuracy : {accuracy} ; Precision : {precision} ; Recall : {recall} ; F1-Score : {f1}")


            logger.info("Model evaluation done...")

        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model...")
        
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__=="__main__":
    trainer = ModelTraining()
    trainer.run()   
