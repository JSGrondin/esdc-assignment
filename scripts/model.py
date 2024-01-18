# Your imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class Model:
    def __init__(self):
        # Instantiate vectorizer object and Multinomial Naive bayes object
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

    def fit(self, X_train, y_train):
        print("Fitting...")
        X_train = self.vectorizer.fit_transform(X_train)
        
        self.classifier.fit(X_train, y_train)

    def predict(self, X_val):
        print("Predicting...")
        X_val = self.vectorizer.transform(X_val)
        predictions = self.classifier.predict(X_val)
        return predictions

