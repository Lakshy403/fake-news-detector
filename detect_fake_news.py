from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model():
    model = BertForSequenceClassification.from_pretrained("./fake_news_model")
    tokenizer = BertTokenizer.from_pretrained("./fake_news_model")
    return model, tokenizer

def predict_fake_news(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Fake News" if prediction == 1 else "Real News"

# Example usage
# model, tokenizer = load_model()
# result = predict_fake_news("Some news text here", model, tokenizer)
# print(result)
