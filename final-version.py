# %%
import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
df = pd.read_csv('clean-data.csv')
df['comment'] = [s.lower() for s in df.comment]
df.drop(['Unnamed: 0'], axis=1, inplace=True)
pd.set_option('display.max_colwidth', None)
df = df.sample(frac=1).reset_index(drop=True)


# %%
df["labels"] = df["class"].map({0: "No offensive text detected", 
                                1: "Possible insult or toxic speech detected"})

# %%
x = df["comment"]
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=100)

# %% [markdown]
# #### Training and modelling

# %%
model = Pipeline([('bow',CountVectorizer()), 
                    ('tfidf',TfidfTransformer()), 
                   # ('tfidfv',TfidfVectorizer()))
                    #('scaler', StandardScaler(with_mean=False)),
                    ('pac',PassiveAggressiveClassifier(C=1, validation_fraction=0.2, shuffle=True, n_jobs=-1))])


# %%
model.fit(X_train, y_train)
model.score(X_train, y_train)

# %%
# Making prediction on test set
y_pred = model.predict(X_test)
  
# Model evaluation
print(f"Test Set Accuracy : {accuracy_score(y_test, y_pred) * 100} %\n\n")  
target = ['no hate', 'hate']
print(f"Classification Report : \n\n{classification_report(y_test, y_pred,target_names=target)}")

# %% [markdown]
# #### Confusion matrix

# %%
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_train, y_train)

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# %% [markdown]
# #### Predicting
# 
# 

# %%
cv = CountVectorizer()
import time


# streamlit config
st.set_page_config(
    page_title="Offensive Speech Classifier",
    page_icon="random",
    layout="centered",
)
st.image('foto.jpeg', width=600)
# %%
def hate_speech_detection():
    st.title("Offensive speech classifier " u"\U0001F3F4\u200D\u2620\uFE0F")
    user = st.text_area("Please insert any text")
    if len(user) < 1:
        st.write("  ")
    else:
        #sample = user
        #data = cv.transform([sample]).toarray()
        a = model.predict([user])
        if a == 'No offensive text detected':
            st.success(a)
        else:
            st.warning(a)
            
hate_speech_detection()