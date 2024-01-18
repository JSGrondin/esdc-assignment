# Your imports
import sys
import os
from turtle import down
project_dir = os.path.dirname(__file__)
sys.path.append(project_dir)
sys.path.append(os.path.dirname(project_dir)) # add parent directory to access utils

from dataset import Dataset, download_nltk_resources
from model import Model
from utils.utils import Results

if __name__ == '__main__':

    # Download required nltk resources
    download_nltk_resources()

    data_path = "../dataset/abstract_arxiv.csv"
    stopwords_path = "../utils/stopwords.txt"

    # Create Dataset object
    data = Dataset(data_path, stopwords_path)

    # Create splits
    X_train, X_val, y_train, y_val = data.create_splits()

    # Create model object
    model = Model()
    
    # Fitting model
    model.fit(X_train, y_train)

    # Predicting on val set
    y_preds = model.predict(X_val)

    # Create results object
    results = Results(y_preds, y_val)

    # Output results and generate heatmaps
    print(f"Accuracy: {results.accuracy()}")
    print(f"Classification Report: \n{results.class_report()}")
    results.conf_mat()
