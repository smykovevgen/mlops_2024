from enum import Enum
from typing import Dict, List, Optional
import os
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_training.exceptions import (
    AlreadyExistsError,
    InvalidData,
    NameKeyError,
    ParamsTypeError,
)
from model_training.model_training import ModelFactory

app = FastAPI(title="MLApp")
allmodels = ModelFactory()


class ClassType(Enum):
    cb = "cb"
    rf = "rf"


class FullType(Enum):
    cb = "<class 'catboost.core.CatBoostClassifier'>"
    rf = "<class 'sklearn.ensemble._forest.RandomForestClassifier'>"


class Model(BaseModel):
    user_model_name: str
    type_model: ClassType
    params: Dict
    fitted: bool


class ModelTypes(BaseModel):
    model_name: ClassType
    model_type: FullType


class Data(BaseModel):
    X: Dict[str, List[float]]
    y: Optional[List[float]]


@app.get("/get_available_model_types", status_code=200)
async def getting_available_model_types() -> List[ModelTypes]:
    return allmodels.get_available_model_types(show=True)


@app.get("/get_models", status_code=200)
async def getting_models(
    only_fitted: bool | None = False,
    all_params: bool | None = False,
    name_model: str | None = None,
) -> List[Model]:
    try:
        return allmodels.get_models(only_fitted, all_params, name_model)
    except NameKeyError as e:
        raise HTTPException(status_code=404, detail=e.txt)


@app.post("/init_new_model", status_code=201)
async def init_models(
    type_model: str,
    user_model_name: str,
    params: Dict = {"random_state": 42, "n_estimators": 100},
) -> Model:
    try:
        return allmodels.init_new_model(
            type_model, user_model_name, params=params
        )
    except AlreadyExistsError as e:
        raise HTTPException(status_code=400, detail=e.txt)
    except NameKeyError as e:
        raise HTTPException(status_code=404, detail=e.txt)
    except ParamsTypeError as e:
        raise HTTPException(status_code=400, detail=e.txt)


@app.put("/model_fit/{user_model_name}", status_code=200)
async def model_fit(user_model_name: str, data: Data) -> Model:
    try:
        list_values = list(data.X.values())
        X = np.zeros((len(list_values[0]), len(data.X)))
        y = np.array(data.y)
        for feature_id in range(len(list(data.X.keys()))):
            X[:, feature_id] = list_values[feature_id]
        allmodels.model_fit(X, y, user_model_name)
        return allmodels.get_models(name_models=user_model_name)[0]
    except NameKeyError as e:
        raise HTTPException(status_code=404, detail=e.txt)
    except InvalidData as e:
        raise HTTPException(status_code=400, detail=e.txt)


@app.put("/model_predict/{user_model_name}", status_code=200)
async def model_predict(user_model_name: str, data: Data) -> Dict[str, List]:
    try:
        list_values = list(data.X.values())
        X = np.zeros((len(list_values[0]), len(data.X)))
        for feature_id in range(len(list(data.X.keys()))):
            X[:, feature_id] = list_values[feature_id]
        return {
            "preds": list(
                allmodels.model_predict(X, user_model_name).flatten()
            )
        }
    except NameKeyError as e:
        raise HTTPException(status_code=404, detail=e.txt)
    except InvalidData as e:
        raise HTTPException(status_code=400, detail=e.txt)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Incorrect data for prediction"
        )


@app.delete("/delete_model/{user_model_name}", status_code=200)
async def delete_model(user_model_name: str) -> List[Model]:
    try:
        allmodels.delete_model(user_model_name)
        return allmodels.get_models()
    except NameKeyError as e:
        raise HTTPException(status_code=404, detail=e.txt)


def start():
    """Launched with `poetry run start` at root level"""
    print(os.environ["HOME"])
    if os.environ.get("DOCKER_MODE") == "1":
        uvicorn.run(
            "rest.controller:app",
            host="0.0.0.0",
            port=8005,
            reload=False
        )
    else:
        uvicorn.run(
            "rest.controller:app", host="127.0.0.1", port=8005, reload=True
        )
