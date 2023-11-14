import pandas as pd
import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def load_pipeline(classifier_name):
    if classifier_name == 'Random Forest':
        pipeline = joblib.load('Model/RandomForestClassifier_recall_0.7513_f1_0.6336.pkl')
    elif classifier_name == 'XGBoost':
        pipeline = joblib.load('Model/XGBClassifier_recall_0.6925_f1_0.6211.pkl')
    else:
        pipeline = joblib.load('Model/GradientBoostingClassifier_recall_0.746_f1_0.6186.pkl')

    return pipeline

def user_input_features():
    """
    Collect the user input, create and dictionary and transform it in a Data Frame
    """
    data_dict = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander("Demographic Info", expanded=True):
            data_dict['customer_gender'] = st.selectbox('Gender', ('Female', 'Male'))
            data_dict['customer_SeniorCitizen'] = st.selectbox('Senior Citizen', ('False', 'True'))
            data_dict['customer_Partner'] = st.selectbox('Partner', ('False', 'True'))
            data_dict['customer_Dependents'] = st.selectbox('Dependents', ('False', 'True'))
            data_dict['customer_tenure'] = st.slider('Tenure', 0, 72, 30)
        with st.expander('Phone Service Info',  expanded=True):
            data_dict['phone_PhoneService'] = st.selectbox('Phone Service', ('False', 'True'))
            data_dict['phone_MultipleLines'] = st.selectbox('Multiple Lines', ('False', 'True'))
            
    with col2:
        with st.expander('Internet Service Info', expanded=True):
            data_dict['internet_InternetService'] = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
            data_dict['internet_OnlineSecurity'] = st.selectbox('Online Security', ('False', 'True'))
            data_dict['internet_OnlineBackup'] = st.selectbox('Online Backup', ('False', 'True'))
            data_dict['internet_DeviceProtection'] = st.selectbox('Device Protection', ('False', 'True'))
            data_dict['internet_TechSupport'] = st.selectbox('Tech Support', ('False', 'True'))
            data_dict['internet_StreamingTV'] = st.selectbox('Streaming TV', ('False', 'True'))
            data_dict['internet_StreamingMovies'] = st.selectbox('Streaming Movies', ('False', 'True'))

    with col3:
        with st.expander('Account Info', expanded=True):
            data_dict['account_Contract'] = st.selectbox('Contract Type', ('One year', 'Month-to-month', 'Two year'))
            data_dict['account_PaperlessBilling'] = st.selectbox('Paperless Billing', ('False', 'True'))
            data_dict['account_PaymentMethod'] = st.selectbox('Payment Method', ('Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)'))
            data_dict['account_Charges_Monthly'] = st.slider('Monthly Charges', 0.0, 200.0, 65.0, 0.1)
            data_dict['account_Charges_Total'] = st.number_input('Total Charges', 0.0, 8684.0, 1500.0, 0.1)

    # calculated model variable
    data_dict['Average monthly spend'] = data_dict['account_Charges_Total'] / data_dict['customer_tenure']
    data_dict['Average monthly Spend - Present monthly spend'] = data_dict['Average monthly spend'] - data_dict['account_Charges_Monthly']

    features = pd.DataFrame(data_dict, index=[0])

    # Define the mapping
    boolean_mapping = {'True': 1, 'False': 0}

    # Apply the mapping to the DataFrame
    features = features.replace(boolean_mapping)

    return features

def results(prediction):
    if prediction == 1:
        st.markdown(f"""
        <h1>RESULT: <span style='color: {'#cc2828'};'>Churn</span></h1>
        <hr style='border: 1px solid {'#FF844B'};'>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <h1>RESULT: <span style='color: {'#FF844B'};'>No Churn</span></h1>
        <hr style='border: 1px solid {'#FF844B'};'>
        """, unsafe_allow_html=True)
    return
    
st.set_page_config(layout='wide')

col1, col2 = st.columns(2)

with col1:
    st.title('**Churn Prediction Web App**')
    st.text('by: Fernando Lacerda')

    st.markdown("[![Title](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/fernando-lacerda-/)")

with col2:
    st.image('https://raw.githubusercontent.com/BrunoRaphaell/challenge_dados_2_ed/main/identidade_visual/Logo%20(1).png')

st.sidebar.title("""
**Select Classifier**
                """)

classifier_name = st.sidebar.selectbox('s', ('Random Forest', 'XGBoost', 'Gradient Boosting'), label_visibility='hidden')

st.subheader(':arrow_down: Insert Costumer information', divider='orange')

pipeline = load_pipeline(classifier_name)

df = user_input_features()

# Button to make the prediction
if st.button("Predict Churn", help='Click on this button to make the prediction'):
    # Make a prediction
    prediction = pipeline.predict(df)
    prediction_proba = pipeline.predict_proba(df)

    col1, col2 = st.columns(2)
    with col1:
        results(prediction[0])

        labels = ['Not Churn', 'Churn']
        churn_prop = prediction_proba[0][1]*100
        fig = go.Figure(go.Indicator(
            mode = "number+gauge",
            gauge = {'axis':{'range':[0, 100], 'ticksuffix': '%'},
                'steps': [
                    {'range': [0,100], 'color': '#262730'}],
                'bar': {'color': ("#FF844B" if churn_prop < 50 else "#cc2828")},
                'threshold': {'line': {'color': "white", 'width': 1}, 'thickness': 1, 'value': 50}},
            value = churn_prop,
            delta = {'reference': 100},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text':"<b>Churn Probability</b><br><span style='color: gray; font-size:0.8em'>%</span>", 'font': {"size": 23, 'color': 'white'}}))
        st.plotly_chart(fig, use_container_width=True)