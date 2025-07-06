#import the dependencies
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")


#Step3 define the file path and data preprocess

# File paths
DATASET_PATH = r"C:\Users\zohaib khan\OneDrive\Desktop\USE ME\dump\zk\clickbait.csv"
df = pd.read_csv(DATASET_PATH)
df.dropna(subset=['title', 'clickbait'], inplace=True)

# SpaCy init
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join([token.lemma_ for token in nlp(text) if not token.is_stop or token.text == 'not'])

df['text'] = df['title'].astype(str).apply(preprocess_text)

#Step4 Tokenisation and padding 

MAX_LEN = 16
VOCAB_SIZE = 20000
EMBED_DIM = 100

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post')
y = df['clickbait'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)


#Step5 Load Glove 

GLOVE_PATH = r"C:\Users\zohaib khan\OneDrive\Desktop\USE ME\dump\zk\glove.6B.100d.txt"

embeddings_index = {}
with open(GLOVE_PATH, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coeffs

word_index = tokenizer.word_index
embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM))
for word, i in word_index.items():
    if i < VOCAB_SIZE:
        embedding_matrix[i] = embeddings_index.get(word, np.random.normal(scale=0.6, size=(EMBED_DIM,)))

#Step 6 define deep learning models

def build_gru():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
        GRU(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_ann():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


#Step 7 train and plot keras

def train_model(model_fn, name):
    print(f"\n{name}")
    model = model_fn()
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=128, verbose=0)

    # Plot
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.show()

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

train_model(build_gru, "GRU Model")
train_model(build_lstm, "LSTM Model")
train_model(build_ann, "ANN Model")


#Step 8 Classical Model

X_text = df['text']
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y, test_size=0.2, random_state=42)

def run_model(model, name):
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', model)
    ])
    pipe.fit(X_train_text, y_train_text)
    y_pred = pipe.predict(X_test_text)
    print(f"\n{name} Report:")
    print(classification_report(y_test_text, y_pred))

run_model(LogisticRegression(max_iter=300), "Logistic Regression")
run_model(MLPClassifier(max_iter=300), "MLP Classifier")
run_model(RandomForestClassifier(n_estimators=100), "Random Forest")

