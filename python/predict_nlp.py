from fastapi import FastAPI, HTTPException, Depends, Body, Request
from pydantic import BaseModel
import re
import json
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import keras

app = FastAPI()

okt = Okt()
tokenizer = Tokenizer()

DATA_CONFIGS = 'C:/zee/python/CLEAN_DATA/data_configs.json'
prepro_configs = json.load(open(DATA_CONFIGS, 'r'))

with open('C:/zee/python/CLEAN_DATA/tokenizer.pickle', 'rb') as handle:
    word_vocab = pickle.load(handle)

prepro_configs['vocab'] = word_vocab
tokenizer.fit_on_texts(word_vocab)
MAX_LENGTH = 8

model = keras.models.load_model('C:/zee/python/my_models')
model.load_weights('C:/zee/python/DATA_OUT/cnn_classifier_kr/weights.h5')


class Review(BaseModel):
    sentence: str


@app.post("/predict/")
async def predict_sentiment(request: Request):
    sentence = request.query_params.get("sentence")

    if not sentence:
        raise HTTPException(status_code=400, detail="Sentence not provided")

    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s ]', '', sentence)
    stopwords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']
    sentence = okt.morphs(sentence, stem=True)
    sentence = [word for word in sentence if not word in stopwords]
    vector = tokenizer.texts_to_sequences(sentence)
    pad_new = pad_sequences(vector, maxlen=MAX_LENGTH)

    predictions = model.predict(pad_new)
    predictions = float(predictions.squeeze(-1)[1])

    if predictions > 0.5:
        return {"probability": predictions, "sentiment": "positive"}
    else:
        return {"probability": 1 - predictions, "sentiment": "negative"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
