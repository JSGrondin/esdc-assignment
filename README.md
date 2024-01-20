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
Running the `main_loop.py` will lead to an accuracy of 77.79%. The resulting heatmap is saved in the `/images` folder. In the top left corner of the heatmap, we observe that the model often misclassifies the different types of astrophysics articles (e.g. astro-ph.CO instead of astro-ph.GA). The nuances between the various types of astrophysics categories may be very hard to detect even for a human classifier and there may also be little value in being able to properly categorizes these sub-classes. A solution for this may be to consolidate these 4 different sub-categories into one single class for the purpose of evaluating accuracy and generating the heat map. This would enable a more fair evaluation of the classifier performance.

### Comparison with other ML models
KNN suffers from the curse of dimensionality. Text data typically involves a high-dimensional feature space, especially when using techniques like bag-of-words or TF-IDF. In such high-dimensional spaces, the concept of distance (e.g. euclidean or others) becomes less meaningul, as the distance between most samples tends to be similar. This diminishes the effectiveness of KNN, which relies on distance metrics to identify the 'k' nearest neighbors. In addition, text data is sparse, meaning that each article only has a small subset of the possible words. Distance calculations in sparse spaces are also not very informative, which is another reasy why KNN may suffer in this example.

On the other hand, the Multinomial NB uses a probabilistic approach (not a distance approach), calculating the probability of an article belonging to a particular class, given the words in it. This approach is often more effective for text data, where the presence or frequency of certain words can strongly indicate the category of the text.

## 2. Government Use-case
It is always helpful to consider all steps of the ML project lifecycle, even if I am not necessarily the person who will be responsible for each of these steps. This list would be a good starting point, not necessarily in exact order and some iterations may be necessary.

- **1. Gather data and become one with it**: I would start by gathering a large dataset of past ROEs with the respective comments section (while compliying with data privacy laws and regulations, particularly with sensitive employment data). I would perform an exploratory analysis of the available metadata (if any) and would extract an initial definition of the various RFS categories from this dataset. I would also compare this list of categories extracted from this dataset with what is perhaps available in the EI program guidelines and labor law to ensure that the list is complete and consistent. 

- **2. Gather information from agents**: the list of different RFS categories could then be refined based on insights and feedback from the EI program agents. I would also try to better understand their needs and pain points. Some examples of questions I would try to address: Are some RFSs more difficult to categorize for them and for what reasons? Which of these categories would need a higher recall or higher specificity? How quickly would they need the model to provide a tentative categorization? We would also need to understand how the API would integrate with the EI program’s systems.

- **3. Define suitable metrics (and loss)**: Based on the above, I would select the metrics that are most adapted and aligned with the needs of the organization. It would also be important to include some fairness metrics to ensure the model is fair with all applicants (e.g. male vs female, english vs french, different ethnic backbrounds, etc)

- **4. Clean Data**: I would implement some functionalities to clean the data to remove irrelevant information, remove duplicate entries, fill missing information and standardize formats, etc.

- **5. Clean and/or gather labels**: If the dataset is not already fully labeled with the RFS, we might need to manually label a subset.

- **6. Split Data**: into train, valid and test datasets. I would use the train set for training, the validation set to evaluate the performance and test different hyperparameter settings and the test set at the end of the project to get a good approximation of what would be the performance on unseen data. Instead of randomly assigning each sample to eiter of the set, I may consider a slightly more sophisticated stratification strategy to ensure that each set receives a balance amount of each RFS categories, an equal proportion of male/female and ethnic backgrounds. While splitting the dataset, I would also be cautious about the risk of introducing leaks. For example, I may consider allocating all ROEs provided by the same employer or related to the same employee in the same set. 

- **7. Select Model(s)**: I would choose appropriate learning models, and include some of the simpler baseline models (e.g. NB, SVM) to understand the incremental gains of using more sophisticated approaches (e.g. BERT, LLMs). I would also look into using NLP techniques to analyse the text in the comments and see if there may be ways to extract relevant features that are indicative of the RFS. 

- **8. Train, optimize and evaluate models**: I would build a pipeline for training the various models selected, optimizing hyperparamets and evaluating the performance.  

- **9. Develop user interface**: We would need to also create a user-friendly interface for EI agents to review the model output categorization, and potientially also provide a feedback mechanism where AI agents can correct misclassifications and provide information for ongoing training.  

- **10. Monitoring, reporting and updates:**
We would need to implement a solution to monitor the model performance over time in order to detect any data distribution shifts. We should also plan to regularly update the model with new data and refine the RFS categories if and when they change. It would also be interesting to track the system's impact in terms of the efficiency and effectiveness gained by the EI program.