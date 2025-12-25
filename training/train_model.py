import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from training.train_utils import DATA_FILE_PATH,MODEL_DIR,MODEL_PATH


data = pd.read_csv('data/car_price_prediction.csv').drop_duplicates().drop(['ID','Model'],axis=1)

data = data.rename(columns={'Prod. year':'Year',
                            'Engine volume':'EngineVolume',
                            'Leather interior':'LeatherInterior',
                            'Fuel type':'FuelType',
                             'Gear box type': 'GearBoxType',
                             'Drive wheels':'DriveWheels',
                             })

data['Levy'] = data['Levy'].replace({'-':np.nan}).astype(float)
data['LeatherInterior'] = data['LeatherInterior'].replace({'Yes':True,'No':False})
data['Mileage'] = data['Mileage'].str.extract(r'([\d.]+)').astype(int)
data['Doors'] = data['Doors'].replace({'04-May':4, '02-Mar':2, '>5':5})

data['EngineVolume'] = data['EngineVolume'].str.lower()
data['Turbo'] = data['EngineVolume'].str.contains('turbo')
data['EngineVolume'] = data['EngineVolume'].str.extract(r'([\d.+])').astype(float)

num_feat = ['Levy', 'Year','EngineVolume', 'Mileage', 'Cylinders', 'Doors', 'Airbags']
cat_feat = ['Manufacturer', 'LeatherInterior','Category', 'FuelType', 'GearBoxType',
       'DriveWheels', 'Wheel', 'Color','Turbo']

for col in cat_feat:
    data = data[col].lower()

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])

pipeline = ColumnTransformer([
    ('num',num_pipeline,num_feat),
    ('cat',OneHotEncoder(),cat_feat)
])


y = data['Price']
X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.3,random_state=42)

rf_reg = RandomForestRegressor(max_features=33, n_estimators=200, random_state=42)
rf_model = Pipeline(steps=[
    ('pipe',pipeline),
    ('rf_reg',rf_reg)
])

rf_model.fit(X_train,y_train)
os.makedirs(MODEL_DIR,exist_ok=True)
joblib.dump(rf_model,MODEL_PATH)