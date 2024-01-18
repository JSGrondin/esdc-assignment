# Your imports
import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset import Dataset

if __name__ == '__main__':

    data_path = "../dataset/abstract_arxiv.csv"
    stopwords_path = "../utils/stopwords.txt"

    # Create Dataset object
    data = Dataset(data_path, stopwords_path)

    # Create splits
    X_train, X_val, y_train, y_val = data.create_splits()

    # TODO :: Complete the main (refer to the assignment.md or devoir.md)
