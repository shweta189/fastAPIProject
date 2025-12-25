from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.dependencies import get_api_key, get_current_user
from app.services.model_services import predict_car_price

router = APIRouter()

class CarFeature(BaseModel):
    levy : float                
    prod_year :  int  
    category: str 
    manufacturer:str
    leather_interior : bool
    fuel_type : str 
    engine_volume : float
    mileage : int
    cylinders : float
    gear_box_type : str 
    drive_wheels : str 
    doors : int
    wheel : str 
    color : str 
    airbags : int              
@router.post('/predict')
def predict_price(car_feat:CarFeature,user=Depends(get_current_user), _ = Depends(get_api_key)):
    predicted_price = predict_car_price(car_feat.model_dump())
    return {"Predicted Car Price" :f":{predicted_price:,.2f}"}