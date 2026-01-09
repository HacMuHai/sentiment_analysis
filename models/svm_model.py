from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

class SVMModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train(self, train_df):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train = self.vectorizer.fit_transform(train_df['text'])
        y_train = train_df['sentiment']
        
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_train, y_train)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0], self.model.predict_proba(X)[0] 

    def evaluate(self, test_df):
        correct_predictions = 0
        total_predictions = len(test_df)

        for index, row in test_df.iterrows():
            predicted_sentiment, sentiment_scores = self.predict(row['text'])
            if predicted_sentiment == row['sentiment']:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        return accuracy