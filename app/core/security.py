from datetime import datetime,timezone,timedelta
from jose import jwt,JWTError
from app.core.config import settings

def create_token(data:dict, exp_min=30):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(exp_min)
    to_encode.update({'exp':expire})
    return jwt.encode(to_encode,settings.JWT_SECRET_KEY,settings.JWT_ALGORITHM)

def verify_token(token:str):
    try:
        payload= jwt.decode(token,settings.JWT_SECRET_KEY,algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None
