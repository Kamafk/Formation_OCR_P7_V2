# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Importation des librairies
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle



# 2. Création de l'objet app
app = FastAPI()

# 3. Chargement du modèle
pickle_in = open("pipeline.pkl","rb")
pipeline = pickle.load(pickle_in)

# 4. Déclaration des features
class inputs_data(BaseModel):
    CODE_GENDER : float
    EXT_SOURCE_1 : float
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    NAME_CONTRACT_TYPE_Cash_loans : float
    NAME_EDUCATION_TYPE_Higher_education : float
    NAME_EDUCATION_TYPE_Secondary___secondary_special : float
    OCCUPATION_TYPE_Drivers : float
    CC_AMT_BALANCE_MEAN : float
    CC_AMT_DRAWINGS_ATM_CURRENT_SUM : float
    CC_AMT_DRAWINGS_CURRENT_MEAN : float
    CC_AMT_RECEIVABLE_PRINCIPAL_MEAN : float
    CC_AMT_RECIVABLE_MEAN : float
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN : float
    CC_CNT_DRAWINGS_ATM_CURRENT_VAR : float
    INSTAL_DPD_MEAN : float
    APPROVED_AMT_DOWN_PAYMENT_MAX : float
    PREV_CODE_REJECT_REASON_XAP_MEAN : float
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN : float
    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN : float
    
    
# 5. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello'}

# 6. Expose the prediction functionality, make a prediction from the passed JSON data and return the predicted value
@app.post('/predict')
def get_probability (data: inputs_data):
    
# 6.1. Get features values from the dashboard
    data = data.dict()
    CODE_GENDER = data['CODE_GENDER']
    EXT_SOURCE_1 = data['EXT_SOURCE_1']
    EXT_SOURCE_2 = data['EXT_SOURCE_2']
    EXT_SOURCE_3 = data['EXT_SOURCE_3']
    NAME_CONTRACT_TYPE_Cash_loans = data['NAME_CONTRACT_TYPE_Cash_loans']
    
    NAME_EDUCATION_TYPE_Higher_education = data['NAME_EDUCATION_TYPE_Higher_education']
    NAME_EDUCATION_TYPE_Secondary___secondary_special = data['NAME_EDUCATION_TYPE_Secondary___secondary_special']
    OCCUPATION_TYPE_Drivers = data['OCCUPATION_TYPE_Drivers']
    CC_AMT_BALANCE_MEAN = data['CC_AMT_BALANCE_MEAN']
    CC_AMT_DRAWINGS_ATM_CURRENT_SUM = data['CC_AMT_DRAWINGS_ATM_CURRENT_SUM']
    
    CC_AMT_DRAWINGS_CURRENT_MEAN = data['CC_AMT_DRAWINGS_CURRENT_MEAN']
    CC_AMT_RECEIVABLE_PRINCIPAL_MEAN = data['CC_AMT_RECEIVABLE_PRINCIPAL_MEAN']
    CC_AMT_RECIVABLE_MEAN = data['CC_AMT_RECIVABLE_MEAN']
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN = data['CC_CNT_DRAWINGS_ATM_CURRENT_MEAN']
    CC_CNT_DRAWINGS_ATM_CURRENT_VAR = data['CC_CNT_DRAWINGS_ATM_CURRENT_VAR']
    
    INSTAL_DPD_MEAN = data['INSTAL_DPD_MEAN']
    APPROVED_AMT_DOWN_PAYMENT_MAX = data['APPROVED_AMT_DOWN_PAYMENT_MAX']
    PREV_CODE_REJECT_REASON_XAP_MEAN = data['PREV_CODE_REJECT_REASON_XAP_MEAN']
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN = data['PREV_NAME_CONTRACT_STATUS_Refused_MEAN']
    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN = data['PREV_NAME_PRODUCT_TYPE_walk_in_MEAN']
    
# 6.2. Get the probability
    prediction = pipeline.predict_proba([[CODE_GENDER, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, 
                                    NAME_CONTRACT_TYPE_Cash_loans, NAME_EDUCATION_TYPE_Higher_education,
                                    NAME_EDUCATION_TYPE_Secondary___secondary_special, 
                                    OCCUPATION_TYPE_Drivers, CC_AMT_BALANCE_MEAN,
                                    CC_AMT_DRAWINGS_ATM_CURRENT_SUM, CC_AMT_DRAWINGS_CURRENT_MEAN, 
                                    CC_AMT_RECEIVABLE_PRINCIPAL_MEAN, CC_AMT_RECIVABLE_MEAN,
                                    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN, CC_CNT_DRAWINGS_ATM_CURRENT_VAR,
                                    INSTAL_DPD_MEAN, APPROVED_AMT_DOWN_PAYMENT_MAX, 
                                    PREV_CODE_REJECT_REASON_XAP_MEAN, 
                                    PREV_NAME_CONTRACT_STATUS_Refused_MEAN, 
                                    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN]])[0,1]
    
    return float(prediction)

# 7. Run the API with uvicorn
# Will run on http://127.0.0.1:8000

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port = 8000)
    
#uvicorn api2:app --reload
## Premier app : Nom du fichier
## Second app : Correspond à app dans le fichier