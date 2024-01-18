# Your Imports


# For all methods in this class, feel free to change (remove/add) the parameters if needed
class Dataset:
    # TODO :: If necessary, add variables that you need to initiate this class
    def __init__(self, filename):
        self.filename = filename

        self._load_data()
        self._pre_process()

    # TODO :: Load the dataset
    def _load_data(self):
        print("Loading the data")

    # TODO :: Complete this method that applies the pre-process steps on the dataset (4 steps maximum).
    def _pre_process(self):
        print("Pre-processing the text data...")

    # TODO :: create a training and validation split of the dataset
    def create_splits(self):
        print("Creating splits")
        return ...

    # TODO :: Return the pre-processed text data from all samples in the dataset
    @property
    def text_data(self):
        return ...

    # TODO :: Return the labels from all samples in the dataset
    @property
    def labels(self):
        return ...
