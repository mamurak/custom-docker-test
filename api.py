import src.pred as pred

from typing import Union
from fastapi import Request, FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/")
async def get_body(request: Request):
    request = await request.json()
    output = pred.main(request['text'], './data/model')
    return {"pred": output}
