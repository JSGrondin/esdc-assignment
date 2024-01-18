# Your imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class Model:
    def __init__(self):
        # Instantiate vectorizer object and Multinomial Naive bayes object
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

    def fit(self, X_train, y_train):
        """
        Fitting the tf-idf vectorizer on the train set, while also vectorizing the train data.
        Then Fit the Multinomial NB classifier. 

        Inputs
        ------
        X_train: pd.core.series.Series
            The train data.
        y_train: pd.core.series.Series
            The train labels.
        """
        print("Fitting...")
        X_train = self.vectorizer.fit_transform(X_train)
        
        self.classifier.fit(X_train, y_train)

    def predict(self, X_val):
        """
        Fitting the tf-idf vectorizer on the train set, while also vectorizing the train data.
        Then Fit the Multinomial NB classifier. 

        Inputs
        ------
        X_val: pd.core.series.Series
            The train data to use to obtain the label prediction.

        Returns
        -------
        predictions: np.ndarray, shape=(#samples,)
            The label being predicted for each sample. 
        """
        print("Predicting...")
        X_val = self.vectorizer.transform(X_val)
        predictions = self.classifier.predict(X_val)
        return predictions

