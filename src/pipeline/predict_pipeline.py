import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            features = features.fillna("").replace({None: ""})
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int,):
        
        # 🔹 Fix: Convert None or empty to empty string
        self.gender = gender or ""
        self.race_ethnicity = race_ethnicity or ""
        self.parental_level_of_education = parental_level_of_education or ""
        self.lunch = lunch or ""
        self.test_preparation_course = test_preparation_course or ""
        self.reading_score = reading_score
        self.writing_score = writing_score      
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)