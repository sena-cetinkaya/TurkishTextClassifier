from urllib.parse import quote_plus
from fastapi import APIRouter, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from model_trainer import clear_and_lemmatize
import pickle

router = APIRouter()
templates = Jinja2Templates(directory="templates")

with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

@router.get("/", response_class=HTMLResponse)
def homepage(request: Request, prediction_label: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_label": prediction_label})

@router.post("/xgboost_classifier")
def xgboost_classifier(word: str = Form(...)):
    new_comment_clean = clear_and_lemmatize(word)
    vector = vectorizer.transform([new_comment_clean])

    pred = model.predict(vector)
    prediction_label = le.inverse_transform(pred)

    print(f"Tahmin edilen duygu: {prediction_label[0]}")

    return RedirectResponse(
        url=f"/?prediction_label={quote_plus(prediction_label[0])}&text={quote_plus(word)}",
        status_code=303
    )


