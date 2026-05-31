from pydantic import BaseModel

class PredictionRequest(BaseModel):

    Inbound_Rate: float

    Outbound_Rate: float

    Inbound_Bandwidth_Utilization: float

    Outbound_Bandwidth_Utilization: float