import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class LSTMModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = None

    def create_model(self, vocab_size, max_length):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_df):
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(train_df['text'])
        
        X_train = self.tokenizer.texts_to_sequences(train_df['text'])
        self.max_length = max(len(x) for x in X_train)
        X_train = pad_sequences(X_train, maxlen=self.max_length)
        
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        y_train = train_df['sentiment'].map(sentiment_map)
        
        self.model = self.create_model(5000, self.max_length)
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    def predict(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length)
        prediction = self.model.predict(padded)[0]
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return sentiment_map[np.argmax(prediction)], prediction 

    def evaluate(self, test_df):
        correct_predictions = 0
        total_predictions = len(test_df)

        for index, row in test_df.iterrows():
            predicted_sentiment, sentiment_scores = self.predict(row['text'])
            if predicted_sentiment == row['sentiment']:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        return accuracy