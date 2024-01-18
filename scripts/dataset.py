# Your Imports
import pandas as pd
import nltk
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, filename, stopwords_path):
        """
        Dataset class for Arxiv articles.

        Inputs
        ------
        filename: string
            Path to dataset csv file.
        stopwords_path: string
            Path to stopwords txt file.
        """
        self.filename = filename
        self.stopwords_path = stopwords_path

        self._load_data()
        self._load_stopwords()
        self._pre_process()

    def _load_data(self):
        """
        Loads the Arxiv articles dataset and prints several information
        about the data.
        """
        print("Loading the data")
        self.df = pd.read_csv(self.filename)
        print(f"Dataset has {len(self.df)} rows.")
        print(f"Dataset has the following columns: {self.df.columns}")
        nans_in_df = {col: self.df[col].isnull().sum() for col in self.df.columns}
        print(f"Dataset has the following count of NaNs: {nans_in_df}")
        print(f"Dataset has {self.df['Category'].nunique()} different categories, namely")
        for category in set(self.df['Category'].tolist()):
            print(f"--{category}")

    def _load_stopwords(self):
        # Loads the stopwords txt file and store as attributes for later use.
        with open(self.stopwords_path, "r") as f:
            stopwords = f.read()
            stopwords = stopwords.split('\n')[:-1]
        self.stopwords = stopwords

    def _pre_process(self):
        # Preprocess the 'Abstract' column, i.e. the text data.
        print("Pre-processing the text data...")
        self.df['Abstract'] = self.df['Abstract'].apply(lambda x: self._process_text(x, self.stopwords))

    @staticmethod
    def _process_text(text, stopwords):
        """
        Processes the input text by performing the following four steps:
        1. Tokenizing text
        2. Removing stopwords
        3. Removing punctuation
        4. Stemming remaining tokens

        Inputs
        ------
        text: string
            Text data string corresponding to one sample.
        stopwords: list
            List of all stopwords.
        
        Return
        ------
        string
            Text data without punctuation and stopwords, lowered and stemmed.
        """
        tokens = word_tokenize(text)
        punctuation = set(string.punctuation)
        stemmer = PorterStemmer()

        processed_text = []
        for word in tokens:
            lower_word = word.lower()
            if lower_word not in stopwords and lower_word not in punctuation:
                stemmed = stemmer.stem(lower_word)
                processed_text.append(stemmed)
        
        return ' '.join(processed_text)    
    
    def create_splits(self):
        """
        Splits the data into training and validation sets (80% train, 20% validation).

        Return
        ------
        X_train: pd.core.series.Series
            Training text data.
        X_val: pd.core.series.Series
            Validation text data.
        y_train: pd.core.series.Series
            Training labels.
        y_val: pd.core.series.Series
            Validation labels.
        """
        print("Creating splits")
        X_train, X_val, y_train, y_val = train_test_split(self.text_data, self.labels, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    @property
    def text_data(self):
        return self.df['Abstract']

    @property
    def labels(self):
        return self.df['Category']
