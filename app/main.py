from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api import routes_auth,routes_predict 
from app.middleware.logging import LoggingMiddleware
from app.core.exception import register_exception_handlers

app = FastAPI(title="Car Price Prediction")

#link middleware
app.add_middleware(LoggingMiddleware)

# collection individual routers
app.include_router(routes_auth.router,tags=['Auth'])
app.include_router(routes_predict.router,tags=['Prediction'])

# monitoring using prometheus
Instrumentator().instrument(app).expose(app)

#add exception handlers
register_exception_handlers(app)