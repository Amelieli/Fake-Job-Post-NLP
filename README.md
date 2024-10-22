# Fake-Job-Post-NLP
Super interesting NLP problem! 
# Fake Job Description Prediction
This dataset comes from a Kaggle competition in 2020, which contains 18K job descriptions out of which about 800 are fake. 
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction \
The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent. 

Natural Language Processing(NLP) problems are always fascinating and challenging. The way text features are handled is so very different from that of tabular data and a solid understanding of NLP features and challenges is the foundation to building a Large Language Model (LLM). \
Here I'll attempt to do some intial exploration and follow two routes:   

First route:
 - After EDA, try using Count Vectorizer and TF-IDF Vectorizer to convert text features into vectors.\
 - Then, leverage a Logistic Regression model as classifier to make predictions using converted text features and other binary features under the Tensorflow framework. The use of Logistic Regression model aims to reduce the level of complexity compared to neural nets and help with the explanability of this problem.

Second route:
 - Try using a pretrained Distiled Bert Model to process the text features, then extract the embedding, along with other binary features to fit a Logistic Regression model .
 - You will see the second route returned a much better model performance and it reached ROC-AUC of 97% on the test dataset.\

PS: Although I'm using the word "embedding" casually, the line of codes that does the work is .last_hidden_state and it is not the same concept as the intitial embeddings under this context. 
