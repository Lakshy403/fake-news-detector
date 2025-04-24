from flask import Flask, render_template, request, jsonify
import torch
import pickle
from transformers import BertTokenizer

app = Flask(__name__, template_folder='./templates')

# Load Model & Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("bert_model.pkl", "rb") as f:
    model = pickle.load(f)
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_fake_news(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(output.logits, dim=1)  # Get probabilities
        prediction = torch.argmax(probabilities, dim=1).item()
    
    return {
        "prediction": "Real News 📰" if prediction == 1 else "Fake News ⚠",
        "probabilities": probabilities.tolist()  # Convert tensor to list
    }


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        prediction = predict_fake_news(news_text)
        return render_template('index.html', prediction=prediction)

# Debugging Route to Test Predictions
@app.route('/test', methods=['GET'])
def test_model():
    sample_statements = [
        "NASA has announced a new Mars mission.",
        "The economy is improving due to tax cuts.",
        "A new study shows coffee is good for health."
    ]
    predictions = {text: predict_fake_news(text) for text in sample_statements}
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
