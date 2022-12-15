"""
# My first app
Here's our first attempt at using data to create a table:
"""
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report



st.title('Customer Churn Prediction')
uploaded_file = st.file_uploader("Upload your Customer Churn data(.csv only)")
if uploaded_file is not None:
    # # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    
    #. Preprocessing Data
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.isnull().sum()
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df[np.isnan(df['TotalCharges'])]
    df[df['tenure'] == 0].index

    df.fillna(df["TotalCharges"].mean())

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']


    def object_to_int(dataframe_series):
        if dataframe_series.dtype == 'object':
            dataframe_series = LabelEncoder().fit_transform(dataframe_series)
            
        return dataframe_series

# Choosing model from a list of models

option = st.selectbox(
    'Choose to predict the model ...',
    (
        '1 Random Forest Classifer',
        '2 SVC - Support Vector Classifer',
        '3 ADABOOST - Adaptive Boosting',
        '4 ANN - Artificial Neural Network',
        '5 XGBoost - Extreme Gradient Boosting'
    ))

st.write('Model selected:', option)




if(st.button('Predict')):

    ind = int(option[0])
    st.markdown("""---""")
    modelName = str(ind)+'.sav';
    model = pickle.load(open(modelName,"rb"))
    ypredict = model.predict(df)
    zero=0
    one=0
    male_zero=0
    female_zero=0
    male_one=0
    female_one=0

    for i in ypredict["churn"]:
        if(i == 0):
            zero+=1
            if(ypredict['gender']=='male'):
                male_zero+=1
            else:
                female_zero+=1
        else:
            one+=1
            if (ypredict['gender'] == 'male'):
                male_one += 1
            else:
                female_one += 1

    result = (zero/(zero+one))*100

    st.header('Result : ')

    st.subheader('Percentage of Customer retains - ~'+result+'%')
    plt.figure(figsize=(6, 6))
    labels = ["Churn: Yes", "Churn:No"]
    values = [one, zero]
    labels_gender = ["F", "M", "F", "M"]
    sizes_gender = [female_one, male_one, female_zero, female_one]
    colors = ['#ff6666', '#66b3ff']
    colors_gender = ['#c2c2f0', '#ffb3e6', '#c2c2f0', '#ffb3e6']
    explode = (0.3, 0.3)
    explode_gender = (0.1, 0.1, 0.1, 0.1)
    textprops = {"fontsize": 15}
    # Plot
    plt.pie(values, labels=labels, autopct='%1.1f%%', pctdistance=1.08, labeldistance=0.8, colors=colors, startangle=90,
            frame=True, explode=explode, radius=10, textprops=textprops, counterclock=True, )
    plt.pie(sizes_gender, labels=labels_gender, colors=colors_gender, startangle=90, explode=explode_gender, radius=7,
            textprops=textprops, counterclock=True, )
    # Draw circle
    centre_circle = plt.Circle((0, 0), 5, color='black', fc='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

    # show plot

    plt.axis('equal')
    plt.tight_layout()
    plt.show()
