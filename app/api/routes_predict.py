from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.dependencies import get_api_key, get_current_user
from app.services.model_services import predict_car_price

router = APIRouter()

class CarFeature(BaseModel):
    Levy : float                
    Year :  int  
    Category: str 
    Manufacturer:str
    LeatherInterior : str 
    FuelType : str 
    EngineVolume : float 
    Mileage : int
    Cylinders : float
    GearBoxType : str 
    DriveWheels : str 
    Doors : int
    Wheel : str 
    Color : str 
    Airbags : int              
@router.post('/predict')
def predict_price(car_feat:CarFeature,user=Depends(get_current_user), _ = Depends(get_api_key)):
    predicted_price = predict_car_price(car_feat.model_dump())
    return {"Predicted Car Price" :f":{predicted_price:.2f}"}