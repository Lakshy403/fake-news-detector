import os
import dash
from dash import dcc, html, Input, Output
from flask import Flask
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Check if model exists, otherwise load default
MODEL_PATH = "./fake_news_model"
if not os.path.exists(MODEL_PATH):
    print("Model folder not found. Using default 'bert-base-uncased'.")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
else:
    print("Loading trained model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Fake news prediction function
def predict_fake_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Fake News" if prediction == 1 else "Real News"

# Flask server setup
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Dashboard layout
app.layout = html.Div([
    html.H1("Fake News Detector Dashboard"),
    dcc.Input(id='news-input', type='text', placeholder='Enter news text...', style={'width': '60%'}),
    html.Button('Check', id='check-button', n_clicks=0),
    html.Div(id='output-div')
])

# Callback function to process input and return prediction
@app.callback(
    Output('output-div', 'children'),
    [Input('check-button', 'n_clicks')],
    [dash.dependencies.State('news-input', 'value')]
)
def update_output(n_clicks, news_text):
    if n_clicks > 0 and news_text:
        result = predict_fake_news(news_text)
        return f"Prediction: {result}"
    return ""

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
