from fastapi import FastAPI
from app.routes.predictor import router as predictor_router
import uvicorn
from app.logger import get_logger


log = get_logger('Main App')

app = FastAPI(
    title="Network Anomaly detection API",
    description="API for real-time Anomaly detection",
    version="1.0.0"
)

# Include your router
app.include_router(predictor_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)