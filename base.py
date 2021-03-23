import itertools
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, log_loss, f1_score

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Loan Classification', page_icon="./f.png")
st.title('Loan Classification')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
## About dataset

This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

| Field          | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| Loan_status    | Whether a loan is paid off on in collection                                           |
| Principal      | Basic principal loan amount at the                                                    |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |

\n

The idea is create a model to predict if the loan has been paid or not.
""")

df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')
st.dataframe(df)

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])

st.write("## Data Visualization")

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
st.pyplot()

bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
st.pyplot()

st.write("### Lets look at the day of the week people get the loan")
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
st.pyplot()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

st.write("The final features that we are gonna use looks like the follow table:")
st.dataframe(Feature)
X = Feature
y = df['loan_status'].values

X= preprocessing.StandardScaler().fit(X).transform(X)

st.write(""" 
## Prediction

You have to choose predict with 4 different models:

- K Nearest Neighbor(KNN)
- Decision Tree
- Support Vector Machine
- Logistic Regression

And their metrics are:

| Algorithm          | Jaccard | F1-score | LogLoss |
|--------------------|---------|----------|---------|
| KNN                | 0.653   | 0.790    | 0.586   |
| LogisticRegression | 0.740   | 0.851    | 0.560   |
| SVM                | 0.722   | 0.838    | 0.543   |
| Decision Tree      | 0.754   | 0.860    | 8.330   |
""")

st.sidebar.header("Values to predict")

principal = st.sidebar.number_input("Amount of loan", min_value=400, step=100)
terms = st.sidebar.selectbox("Terms (one, two or four weeks)", [7, 15, 30])
age = st.sidebar.slider("Age", 18, 90, 18)
gender = st.sidebar.selectbox("Gender", [ "Female","Male",])
gender_bin = 0 if gender == "Male" else 1
effective_date = st.sidebar.date_input("Date")
effective_date = pd.to_datetime(effective_date)
day_of_week = effective_date.dayofweek
weekend = 1 if (day_of_week>3) else 0
education_labels = ["High School or Below", "Bachelor", "college"]
education = st.sidebar.selectbox("Education", education_labels)
education_dummies = pd.DataFrame([[0,0,0]], columns=education_labels)
education_dummies[education] = 1

user_df = pd.DataFrame([[principal, terms, age, gender_bin, weekend]], columns=["Principal", "Terms", "Age", "Gender", "Weekend"])
user_df = pd.concat([user_df, education_dummies], axis=1)
st.write("---")
st.write("**User Dataframe**. You can change the values in the left sidebar ")
st.dataframe(user_df)
X_ = preprocessing.StandardScaler().fit_transform(user_df.to_numpy()[0][:, np.newaxis]).reshape(1,-1)

st.sidebar.write("## Algorithm to use")
algo = st.sidebar.selectbox("Select", ["KNN", "LogisticRegression", "Tree", "SVM"])

if algo == "KNN":
  model = pickle.load(open("KNN.pkl", "rb"))
elif algo == "LogisticRegression":
  model = pickle.load(open("LR.pkl", "rb"))
elif algo == "Tree":
  model = pickle.load(open("DT.pkl", "rb"))
else:
  model = pickle.load(open("SVC.pkl", "rb"))

prediction = model.predict(X_)
predict_proba = model.predict_proba(X_)
st.write("**Results**")
col1, col2 = st.beta_columns(2)
col1.write("Prediction")
col1.dataframe(pd.DataFrame(pd.Series(prediction), columns=["Value"]))
col2.write("Probability")
col2.dataframe(pd.DataFrame(model.predict_proba(X_), columns=["Default", "Paidoff"]))


# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/)TODO
""")
# / This app repository