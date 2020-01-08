#natural language processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#delimiter is there to tell that its a tsv file
#quoting=3,means we are ignoring double quotes

#cleaning texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    #re.sub()-to have the things that we want to include
    
    review=review.lower()
    #it is to bring all the uppercase letters to lowercase
    
    review=review.split() #converts string to list
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #set traversal is always faster than list. so chose it
    review=' '.join(review)  #coverts list to string
    corpus.append(review)
    
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()

y=data.iloc[:,1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#fitting the model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

#the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
