import math
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

class NaiveBayesModel:
    def __init__(self):
        self.word_count_positive = defaultdict(int)
        self.word_count_negative = defaultdict(int)
        self.word_count_neutral = defaultdict(int)
        self.log_prior_positive = 0
        self.log_prior_negative = 0
        self.log_prior_neutral = 0
        self.log_likelihood_positive = {}
        self.log_likelihood_negative = {}
        self.log_likelihood_neutral = {}

    # Preprocessing function
    def preprocess_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = tweet.translate(str.maketrans("", "", string.punctuation))
        tokens = tweet.split()
        stemmer = PorterStemmer()
        stopwords_set = set(stopwords.words("english"))
        tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
        return tokens

    # Step 2: Calculate word counts for a given sentiment
    def calculate_word_counts(self, tweets):
        word_count = defaultdict(int)
        for tweet in tweets:
            tokens = self.preprocess_tweet(tweet)
            for token in tokens:
                word_count[token] += 1
        return word_count

    # Step 3: Calculate likelihood using Laplacian smoothing
    def calculate_likelihood(self, word_count, total_words, laplacian_smoothing=1):
        likelihood = {}
        vocabulary_size = len(word_count)

        for word, count in word_count.items():
            likelihood[word] = (count + laplacian_smoothing) / (total_words + laplacian_smoothing * vocabulary_size)

        return likelihood

    def calculate_log_prior(self, sentiment, data):
        return math.log(len(data[data['sentiment'] == sentiment]) / len(data))

    # Step 4: Calculate log prior
    def train(self, train_df):
        # Calculate word counts
        self.word_count_positive = self.calculate_word_counts(train_df[train_df['sentiment'] == 'positive']['text'])
        self.word_count_negative = self.calculate_word_counts(train_df[train_df['sentiment'] == 'negative']['text'])
        self.word_count_neutral = self.calculate_word_counts(train_df[train_df['sentiment'] == 'neutral']['text'])

        # Calculate total words
        total_positive_words = sum(self.word_count_positive.values())
        total_negative_words = sum(self.word_count_negative.values())
        total_neutral_words = sum(self.word_count_neutral.values())

        # Calculate likelihoods
        likelihood_positive = self.calculate_likelihood(self.word_count_positive, total_positive_words)
        likelihood_negative = self.calculate_likelihood(self.word_count_negative, total_negative_words)
        likelihood_neutral = self.calculate_likelihood(self.word_count_neutral, total_neutral_words)

        # Calculate log likelihoods
        self.log_likelihood_positive = {word: math.log(prob) for word, prob in likelihood_positive.items()}
        self.log_likelihood_negative = {word: math.log(prob) for word, prob in likelihood_negative.items()}
        self.log_likelihood_neutral = {word: math.log(prob) for word, prob in likelihood_neutral.items()}

        # Calculate log priors
        self.log_prior_positive = self.calculate_log_prior('positive', train_df)
        self.log_prior_negative = self.calculate_log_prior('negative', train_df)
        self.log_prior_neutral = self.calculate_log_prior('neutral', train_df)

    # Step 5: Predict sentiment for a given tweet
    def predict(self, text):
        tokens = self.preprocess_tweet(text)

        log_score_positive = self.log_prior_positive + sum([self.log_likelihood_positive.get(token, 0) for token in tokens])
        log_score_negative = self.log_prior_negative + sum([self.log_likelihood_negative.get(token, 0) for token in tokens])
        log_score_neutral = self.log_prior_neutral + sum([self.log_likelihood_neutral.get(token, 0) for token in tokens])

        sentiment_scores = {
            'positive': log_score_positive,
            'negative': log_score_negative,
            'neutral': log_score_neutral
        }

        predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        return predicted_sentiment, sentiment_scores 

    # Step 6: Evaluate the model on the test set
    def evaluate(self, test_df):
        correct_predictions = 0
        total_predictions = len(test_df)

        for index, row in test_df.iterrows():
            predicted_sentiment, sentiment_scores = self.predict(row['text'])
            if predicted_sentiment == row['sentiment']:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {accuracy}")
        return accuracy