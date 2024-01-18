# esdc-assignment
Machine Learning Home Test for ESDC Chief Data Office

## 1. Analysis 
### Data
- ***Data exploration***  : I looked at whether there were NaNs in the data and saw there were none. I confirmed the number of samples and columns. I also looked at the number of different categories in the labels and the count for each and saw that there were all balanced with around 500 each. I also looked at the statistics in terms of the length
of the sample's text data (min: 16, mean: 936 and max: 2273 characters).

- ***Pre-processing*** : The following steps have been selected:
    - *tokenization*: Breaks down the text into individual words or tokens. This is essential to convert the raw text into the fundamental units for further processing.
    - *removing stopwords*: commonly used words such as 'the', 'is' 'in' are removed from the text as they generally do not carry significant meaning and are removed to reduce the dataset size and complexity. This helps the model focus on the words that carry more importance for the categorization task.
    - *removing punctuation*: typically not useful in understanding the meaning of abstracts. Removing them symplifies the text data and can improve performance. 
    - *stemming*: reduces the words to their root form (e.g. 'running', 'runs' and 'ran' are all reduced to 'run'). Helps in reducing the complexity of the text data by consolidating different forms of a word into a common base form. 
    - *tf-idf vectorization*: although not included in the Dataset class, this is another important pre-processing step. This technique enables converting the text data into numerical values (as an alternative to using a Bag of Words approach) that highlight the importance of words within each abstract description relative to the whole dataset. Words that appear frequently in few abstracts but rarely in other documents are more significant. 

- ***Impact in terms of computation and model behavior of training model without pre-processing***: Pre-processing steps like removing stopwords and punctuation, lowering the text, and stemming are crucial for focusing the model's learning on relevant features. Whithout these steps, the model might learn from noise and irrelevant details. Training on unprocessed data with many irrelevant features can lead to overfitting, where the model learns the training data very well but performs poorly on new data. In terms of computation, these pre-processing steps contributes to reducing the feature sapce (i.e. the pre-processed vocab is small than the raw version) which means pre-processing may lead to faster training times and lower memory usage. 


### Histogram
An unbalanced distribution of the labels can introduce a bias during the learning process that will favor the majority class or the class with more samples, and vice versa for the classes with fewer samples. This can lead to poor predictive performance of the minority class. This also means that some metrics like accuracy will be misleading, as a high accuracy could be obtained by simply predicting the majority class for all instances in severe cases. 

In terms of what can be done, among other things:
- some metrics would be more appropriate like precision, recall, F1-Score instead of the accuracy for example. 
- some learning algorithms enable modifying the weight of the minority class to give it more importance during training and to compensate for the imbalanced problem.
- it is also possible to oversample the minority class, or undersample the majority class. 

### Results
x

### Comparison with other ML models
x

## 2. Government Use-case