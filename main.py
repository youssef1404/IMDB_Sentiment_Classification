from fastapi import FastAPI
from utils import predict_new

# Intialize the app
app = FastAPI(title='IMDB_Review')

# Endpoint for healthy check
@app.get('/', tags=['General'])
async def home():
    return {'up & running'}


# Endpoint for Prediction
@app.post('/predict', tags=['IMDB_Sentiment'])
async def imdbReview(data):
    
    # Call the function from utils.py
    pred = predict_new(data)

    return {f"Prediction is: {pred}"}