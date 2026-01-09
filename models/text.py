
# Add SVM Model
def train_svm_model(train_df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['sentiment']

    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    return svm_model, vectorizer


def predict_svm(text, model, vectorizer):
    X = vectorizer.transform([text])
    return model.predict(X)[0], model.predict_proba(X)[0]

# Add LSTM Model


def create_lstm_model(vocab_size, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_lstm_model(train_df):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_df['text'])

    X_train = tokenizer.texts_to_sequences(train_df['text'])
    max_length = max(len(x) for x in X_train)
    X_train = pad_sequences(X_train, maxlen=max_length)

    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y_train = train_df['sentiment'].map(sentiment_map)

    model = create_lstm_model(5000, max_length)
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    return model, tokenizer, max_length


def predict_lstm(text, model, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded)[0]
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_map[np.argmax(prediction)], prediction
