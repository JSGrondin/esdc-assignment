import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class Results:
    def __init__(self, predictions, y_val):
        self.predictions = predictions
        self.y_val = y_val
        self.target_names = None

    def accuracy(self):
        acc_score = accuracy_score(self.y_val, self.predictions)
        return acc_score

    def class_report(self):
        self.target_names = sorted(list(set(self.y_val)))
        c_r = classification_report(self.y_val, self.predictions, target_names=self.target_names)
        return c_r

    def conf_mat(self):
        c_m = confusion_matrix(self.y_val, self.predictions)
        df_cm = pd.DataFrame(c_m, columns=[self.target_names], index=[self.target_names])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()


def remove_non_ascii(word):
    new_word = []
    for char in word:
        new_char = unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_word.append(new_char)
        if char != new_char:
            print(f"{char} -> {new_char}")
    return ''.join(new_word)
