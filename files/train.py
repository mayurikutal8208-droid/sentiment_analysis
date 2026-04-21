import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  ## Converts text labels into numerical labels
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  ## Makes all sequences of the same length
from preprocessing import clean_text

df = pd.read_csv('data/twitter_training.csv')

df.columns = ["ID", "Topic", "Sentiment", "Text"]

df = df[["Text", "Sentiment"]]

print(df.head())

## Cleaning

df['clean_text'] = df['Text'].apply(clean_text)

## Encoding Labels

le = LabelEncoder()

df['label'] = le.fit_transform(df['Sentiment'])

##Save the label encoding

joblib.dump(le, "models/label_encoder.pkl")

## Tokenization

tokenizer = Tokenizer(num_words=10000)  ## Only consider the top 10000 words in the dataset

tokenizer.fit_on_texts(df['clean_text'])

X = tokenizer.texts_to_sequences(df['clean_text'])

X = pad_sequences(X, maxlen=100)

## Save tokenizer

joblib.dump(tokenizer, "models/tokenizer.pkl")

y = df['label']

## Split the data into train-test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num_classes = len(le.classes_)

## LSTM model

model_lstm = Sequential(

    [

        Embedding(input_dim=10000, output_dim=128, input_length=100),   ## every word becomes a 128 dim vector

        LSTM(64), ## this is the LSTM layer with 64 units

        Dense(32, activation='relu'),  #Hidden layer for learning patterns

        Dense(num_classes, activation='softmax')  ## Output layer with 3 classes and softmax activation

    ]

)

model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

model_lstm.save("models/lstm_model.h5")

## GRU model

model_gru = Sequential(

    [

        Embedding(input_dim=10000, output_dim=128, input_length=100),   ## every word becomes a 128 dim vector

        GRU(64), ## this is the GRU layer with 64 units

        Dense(32, activation='relu'),  #Hidden layer for learning patterns

        Dense(num_classes, activation='softmax')  ## Output layer with 3 classes and softmax activation

    ]

)

model_gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_gru.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

model_gru.save("models/gru_model.h5")

print("Training complete. Models saved")