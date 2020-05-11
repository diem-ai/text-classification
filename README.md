# Text Classification with TF-IDF, Word Embedding and Naive Bayes

- This project introduces one of techniques of natural language processing: text classification. We shall explore step by step how to create new features through data analysis and do data cleaning. We also explain the conception of Term Frequency - Inverse Document Frequency and Word Embedding and how to implement them through the real data. 

- The article will show you how to apply Naives Bayes on text data after the tokenization and vectorization.  We will use one of Selection Best Feature techniques to chose only features that contribute to the performance of the prediction. After that, we will analyse and compare the evaluation metrics including accuracy , confusion matrix and classification report between baseline accuracy, TF-IDF + Bayes and Word Embedding + Bayes.

- Check out the blog spot for more detail: [Text Classification with TF-IDF, Word Embedding and Naive Bayes](https://diem-ai.blogspot.com/2020/05/text-classification-with-tf-idf-word.html)

## The project covers:

### <code>spam_EDA.ipynb</code>: Data Exploratory Analysis: 
- The work will anwser the quesions:
1) What percentage of the documents are spam?
2) What is the longest message ?
3) What is the average length of documents (number of characters) for not spam and spam documents?
4) What is the average number of digits per document for not spam and spam documents?
5) What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
- As the result, 3 new features are created: len_text', 'digits', 'non_alpha_char'

### <code>NativeBayes_Tf-Idf.ipynb</code>: Transform text data into Term Frequency - Inverse Document Frequency, select the best feature with f_classif and fit the transformed data to Bayesian algorithm: 
- The accurary score is 91%. It is 5% better than the baseline.
- From Classification Report:
  - The F1-score (spam class) is 74% and F1-score (Not Spam class) is 95%
- From Confusion Matrix:
  - The correction of prediction = True Positive (TP) + True Negative (TN) = 878 + 140 = 918
  - The misclassification = False Negative (FN) + False Positive (FP) = 84 + 13 = 97

### <code>NativeBayes_word2vec.ipynb</code>Transform text data into Word Embedding, select the best feature with f_classif and fit the transformed data to Bayesian algorithm: 
- The accurary score is 94%. It is 8% better than the baseline (86%).
- From Classification Report:
  - The F1-score (spam class) is 80% and F1-score (Not Spam class) is 96%
- From Confusion Matrix:
  - The correction of prediction = TP + TN = 896 + 147 = 1043
  - The mis-classification = FN + FP = 84 + 13 = 72

## The outcome:
1) TF-IDF + Naives Bayes: improve 5% of accuracy of classification from 86% to 91%
2) Word2Vec + Naive Bayes: improve 8% of accuracy of classification from 86% to 94%

### Requirements
Python >= 3.7
Jupyter Notebook

### Dependencies
requirement.txt

## Run the Notebook on the local:
- Checkout the project : git clone https://github.com/diem-ai/text-classification.git
- Install the latest version of libraries in requirements and dependencies
- Run the following commands in order:
1) jupyter notebook spam_EDA.ipynb
2) jupyter notebook NativeBayes_Tf-Idf.ipynb
3) jupyter notebook NativeBayes_word2vec.ipynb



