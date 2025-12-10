import joblib

# Load trained model
model = joblib.load("model/sentiment_model.pkl")

print("Movie Review Sentiment Analysis (Real-Time)")
print("Type 'exit' to quit.\n")

while True:
    text = input("Enter a movie review: ")
    
    if text.lower() == "exit":
        break
    
    prediction = model.predict([text])[0]
    
    print("Prediction:", prediction)
    print("-" * 40)
