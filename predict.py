import joblib

# ---------------------------
# LOAD MODEL
# ---------------------------

model = joblib.load(
    "models/fake_news_model.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

print("="*50)
print("AI FAKE NEWS DETECTOR")
print("="*50)

while True:

    text = input(
        "\nPaste News Article (or type exit):\n"
    )

    if text.lower() == "exit":
        break

    text_vector = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        text_vector
    )[0]

    confidence = abs(
        model.decision_function(
            text_vector
        )[0]
    )

    if prediction == 0:
        print("\n❌ FAKE NEWS")
    else:
        print("\n✅ REAL NEWS")

    print(
        f"Confidence Score: {confidence:.2f}"
    )