import pandas as pd
import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from helpers import *

COLOR_MAP = {
    0: '#872b95',
    1: '#FF844B'  
    }      

def distribution_plot_plotly(target_variable, independent_variables, dataframe, subplot_title, y_limit=4900, height=1000, width=1200):

    # Calculate the number of rows and columns for subplots
    num_vars = len(independent_variables)
    num_cols = math.ceil(math.sqrt(num_vars))
    num_rows = math.ceil(num_vars / num_cols)

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[col[1] for col in independent_variables])

    # Iterate through each variable and create a plot
    for i, (col_name, title) in enumerate(independent_variables, start=1):
        # Count the occurrences for each category
        count_data = dataframe.groupby([col_name, target_variable]).size().reset_index(name='count')

        # Add traces
        for churn_value in dataframe[target_variable].unique():
            subset = count_data[count_data[target_variable] == churn_value]
            row, col = ((i - 1) // num_cols) + 1, ((i - 1) % num_cols) + 1
            fig.add_trace(
                go.Bar(x=subset[col_name], y=subset['count'], name=f'Churn: {churn_value}', marker_color=COLOR_MAP[churn_value]),
                row=row, col=col
            )

        # Update layout for each subplot
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
        fig.update_xaxes(title_text='Churn', row=row, col=col)

    # Update overall layout
    fig.update_layout(height=height, width=width, title_text=subplot_title, title_font=dict(size=30, family='Arial, bold'), showlegend=False, yaxis=dict(range=[0, y_limit]))

    return fig

def histogram_numeric_variables(independent_variables, dataframe):
    fig = make_subplots(rows=2, cols=len(independent_variables), 
                    vertical_spacing=0.1, 
                    row_heights=[0.8, 0.2],  # 4:1 ratio
                    subplot_titles=[col[1] for col in independent_variables] + ['' for _ in independent_variables])

    for i, (col, title) in enumerate(independent_variables, start=1):
            # Add histogram to the first row
            fig.add_trace(
                go.Histogram(
                    x=dataframe[col], 
                    name=title, 
                    marker=dict(
                        color='rgba(255, 113, 49, 1)',  # Lighter fill color with transparency
                        line=dict(color='#1B1212', width=2)  # Highlighted border
                    ),
                    nbinsx=30
                ),
                row=1, col=i
            )

            # Add boxplot to the second row
            fig.add_trace(
                go.Box(x=dataframe[col], name=title, marker_color='#ff7131'),
                row=2, col=i
            )

            # Remove y-axis labels for boxplots
            fig.update_yaxes(title_text="", showticklabels=False, row=2, col=i)

    # Update layout
    fig.update_layout(height=500, width=1500, title_text="Distribution of Numeric Variables", title_font=dict(size=30, family='Arial, bold'), showlegend=False)
    return fig

st.set_page_config(layout='wide')

col1, col2 = st.columns(2)

with col1:
    st.title('**Exploratory Data Analysis**')
    st.text('by: Fernando Lacerda')

    st.markdown("[![Title](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/fernando-lacerda-/)")

with col2:
    st.image('https://raw.githubusercontent.com/BrunoRaphaell/challenge_dados_2_ed/main/identidade_visual/Logo%20(1).png')

df_viz = pd.read_csv('data/churn_data.csv')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Churn', 'Demographic info', 'Phone Service', 'Internet Service', 'Numeric Features'])

with tab1:
    fig = px.histogram(df_viz, x="Churn", title='Churn Distribution', color='Churn', color_discrete_map=COLOR_MAP)
    fig.update_layout(bargap=0.2, xaxis_title='Churn', yaxis_title='Frequency', xaxis_showticklabels=True,  title_font=dict(size=30, family='Arial, bold'))

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    demographic_columns = [
        ('customer_gender', 'Gender'), 
        ('customer_SeniorCitizen', 'Senior Citizen'), 
        ('customer_Partner', 'Partner'), 
        ('customer_Dependents', 'Dependents')
    ]

    fig = distribution_plot_plotly('Churn', demographic_columns, df_viz, subplot_title="Effect of Demographic Variables on Churn", height=700, width=900)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    phone_columns = [('phone_MultipleLines', 'Multiple Lines'), ('phone_PhoneService', 'Phone Service')]
    fig = distribution_plot_plotly('Churn', phone_columns, df_viz, subplot_title="Effect of Phone Service Variables on Churn", height=500, width=600)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    internet_columns = [('internet_InternetService', 'Internet Service'), ('internet_OnlineSecurity', 'Multiple Online Security'), ('internet_OnlineBackup', 'Online Backup'), ('internet_DeviceProtection', 'Device Protection'),
                ('internet_TechSupport', 'Tech Support'), ('internet_StreamingTV', 'Streaming TV'), ('internet_StreamingMovies', 'Streaming Movies')]
    
    fig = distribution_plot_plotly('Churn', internet_columns, df_viz, subplot_title="Effect of Internet Service Variables on Churn", height=1200, width=1500, y_limit=4000)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    numeric_columns = [('customer_tenure', 'Tenure'), ('account_Charges_Monthly', 'Monthly Charge'), ('account_Charges_Total', 'Life time total spent')]
    fig = histogram_numeric_variables(numeric_columns, df_viz)
    st.plotly_chart(fig, use_container_width=True)
