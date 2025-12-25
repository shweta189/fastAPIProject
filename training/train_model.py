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


data = pd.read_csv(DATA_FILE_PATH).drop_duplicates().drop(['ID','Model'],axis=1)

data = data.rename(columns={'Price':'price',
                            'Levy':'levy',
                            'Prod. year':'prod_year',
                            'Engine volume':'engine_volume',
                            'Mileage':'mileage',
                            'Cylinders':'cylinders',
                            'Doors':'doors',
                            'Airbags':'airbags',
                            'Manufacturer':'manufacturer',
                            'Leather interior':'leather_interior',
                            'Category':'category',
                            'Fuel type':'fuel_type',
                             'Gear box type': 'gear_box_type',
                             'Drive wheels':'drive_wheels',
                             'Wheel':'wheel',
                             'Color':'color'
                             })

data['levy'] = data['levy'].replace({'-':np.nan}).astype(float)
data['leather_interior'] = data['leather_interior'].replace({'Yes':True,'No':False})
data['mileage'] = data['mileage'].str.extract(r'([\d.]+)').astype(int)
data['doors'] = data['doors'].replace({'04-May':4, '02-Mar':2, '>5':5})

# data['engine_volume'] = data['engine_volume'].str.lower()
# data['turbo'] = data['engine_volume'].str.contains('turbo')
data['engine_volume'] = data['engine_volume'].str.extract(r'([\d.+])').astype(float)



num_feat = ['levy', 'prod_year','engine_volume', 'mileage', 'cylinders', 'doors', 'airbags']
cat_feat = ['manufacturer', 'leather_interior','category', 'fuel_type', 'gear_box_type',
       'drive_wheels', 'wheel', 'color']

print(list(zip(data.columns ,data.dtypes)))
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])


pipeline = ColumnTransformer([
    ('num',num_pipeline,num_feat),
    ('cat',OneHotEncoder(),cat_feat)
])


y = data['price']
X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.3,random_state=42)

rf_reg = RandomForestRegressor(max_features=33, n_estimators=200, random_state=42)
rf_model = Pipeline(steps=[
    ('pipe',pipeline),
    ('rf_reg',rf_reg)
])

rf_model.fit(X_train,y_train)
os.makedirs(MODEL_DIR,exist_ok=True)
joblib.dump(rf_model,MODEL_PATH)