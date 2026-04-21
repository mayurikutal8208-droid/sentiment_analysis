import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import clean_text

## Load the saved models and artifacts

tokenizer = joblib.load("models/tokenizer.pkl")

label_encoder = joblib.load("models/label_encoder.pkl")

lstm_model = load_model("models/lstm_model.h5")

gru_model  = load_model("models/gru_model.h5")

def predict(text, model_type = "lstm"):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(seq, maxlen=100)

    if model_type == "lstm":

        pred = lstm_model.predict(padded)

    else:

        pred = gru_model.predict(padded)

    label = label_encoder.inverse_transform([pred.argmax()])[0]

    return label

if __name__ == "__main__":

    test_text = "I love this product! It's amazing and I found it very useful for my usage."

    print("LSTM Prediction:", predict(test_text, model_type="lstm"))

    print("GRU Prediction:", predict(test_text, model_type="gru"))
