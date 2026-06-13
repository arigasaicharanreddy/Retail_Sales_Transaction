from pydantic import BaseModel 

class Amount(BaseModel):
    Quantity: int
    Price: float
    DiscountApplied_Percent: float  
    ProductCategory: str            
    PaymentMethod: str              