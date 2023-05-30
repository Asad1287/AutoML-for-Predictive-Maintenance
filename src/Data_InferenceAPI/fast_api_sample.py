from fastapi import FastAPI
from typing import Any, List
from fastapi import FastAPI
from typing import Any, List
import pandas as pd
import uvicorn
from MongoHandler import * 
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Item(BaseModel):
    UDI: Optional[int]
    Type: Optional[int]
    Air_temperature: Optional[float]
    Process_temperature: Optional[float]
    Rotational_speed: Optional[int]
    Torque: Optional[float]
    Tool_wear: Optional[int]
    TWF: Optional[int]
    HDF: Optional[int]
    PWF: Optional[int]
    OSF: Optional[int]
    RNF: Optional[int]
    predictions: Optional[float]

app = FastAPI()

PASSWORD = "1234"
INFERENCE_COLLECTION = "inference_results"
DATABASE = "predictive_maintenance"

client = MongoDBHandler(f"mongodb+srv://root12345:{PASSWORD}@cluster1.b03tix4.mongodb.net/",f"{DATABASE}",f"{INFERENCE_COLLECTION}")

@app.on_event("startup")
def load_data():
    global df
    df = client.read_to_pandas()
    if "_id" in df.columns:
        df["_id"] = df["_id"].apply(lambda x: str(x))
    df['predictions'] = df['predictions'].apply(lambda x: 1 if x > 0.5 else 0)

@app.get("/")
async def root():
    return {"message": "Welcome to the API, proceed to /data"}

@app.get("/data")
async def get_data():
    return df.to_dict(orient='records')

@app.get("/data/{product_id}")
async def get_data_by_product_id(product_id: int):
    data = client.read_single({'Product ID': product_id})
    if data:
        return data
    else:
        raise HTTPException(status_code=404, detail="Item not found")

@app.put("/data/{product_id}")
async def update_item(product_id: int, item: Item):
    if client.update({'Product ID': product_id}, {"$set": item.dict()}):
        return {"message": "Data has been updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/data/{product_id}")
async def delete_item(product_id: int):
    if client.delete({'Product ID': product_id}):
        return {"message": "Data has been deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
