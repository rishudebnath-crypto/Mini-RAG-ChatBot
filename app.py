from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
from Mini_RAG import get_response
import json

app = FastAPI()

class Query(BaseModel):

    question: Annotated[str, Field(..., description='Enter your query here')]

@app.get('/')
def home_page():
    return {'message': 'Welcome!'}

@app.post('/chat')
def chat(query: Query):
    result = get_response(query.question)
    return {'response': result}

@app.get('/history')
def get_history():
    with open('chat_history.json', 'r') as f:
        data = json.load(f)

    return data

@app.delete('/delete')
def delete_history():
    data = []
    with open('chat_history.json', 'w') as f:
        json.dump(data, f, indent=4)

    return {'message': 'Chat History deleted succesfully'}
