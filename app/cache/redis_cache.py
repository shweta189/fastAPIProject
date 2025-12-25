import redis 
import json
from app.core.config import settings

redis_client = redis.Redis.from_url(settings.REDIS_URL)


def get_cached_prediction(key:str):
    value = redis_client.get(key)
    if value:
        return eval(value)
    return None

def set_cached_prediction(key:str,value:dict,exp:int=3600):
    redis_client.set(key,str(value))