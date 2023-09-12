import requests
import json
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

color_palette = ['#171821', '#872b95', '#ff7131', '#fe3d67']
binary_palette = ['#872b95', '#fe3d67']

def api_request(url):

    response = requests.get(url)

    if response.status_code == 200:
        json = response.json()
    else:
        print('An unexpected error occured')
    return response.json()

def flatten(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def encoder(dataframe, columns):
    encoded_list = []
    for col in columns:
        encoder_instance = OneHotEncoder(drop='first')
        encoded_array = encoder_instance.fit_transform(dataframe[[col]]).toarray()
        encoded_df = pd.DataFrame(encoded_array, columns=encoder_instance.get_feature_names_out([col]))
        encoded_list.append(encoded_df)

    dataframe = dataframe.drop(columns=columns)
    dataframe = pd.concat([dataframe] + encoded_list, axis=1)
    
    return dataframe


def label_bars(ax, vertical=True, horizontal_padding=0, vertical_padding=5):
    """
    Annotate the bars with their respective heights (for vertical bars) or widths (for horizontal bars)
    and include the percentage in parentheses.

    Parameters:
    - ax: The axis object containing the bars.
    - vertical: Whether the bars are vertical. If False, it's assumed they are horizontal.
    - horizontal_padding: Horizontal padding for the label position.
    - vertical_padding: Vertical padding for the label position.
    """
    # Calculate total (either sum of all heights for vertical bars or widths for horizontal bars)
    total = sum(p.get_height() for p in ax.patches) if vertical else sum(p.get_width() for p in ax.patches)
    
    for p in ax.patches:
        if vertical:
            label_value = p.get_height()
            percentage = 100 * label_value / total
            ax.annotate(f'{int(label_value)} ({percentage:.1f}%)', 
                        (p.get_x() + p.get_width() / 2., label_value), 
                        ha='center', va='center', 
                        xytext=(horizontal_padding, vertical_padding), 
                        textcoords='offset points')
        else:
            label_value = p.get_width()
            percentage = 100 * label_value / total
            ax.annotate(f'{int(label_value)} ({percentage:.1f}%)', 
                        (label_value, p.get_y() + p.get_height() / 2.), 
                        ha='center', va='center', 
                        xytext=(horizontal_padding, vertical_padding), 
                        textcoords='offset points')

def distribution_plot(target_variable, independent_variables, dataframe, y_limit=4900):
    
    # Calculate the number of rows and columns for subplots
    num_vars = len(independent_variables)
    num_cols = math.ceil(math.sqrt(num_vars))
    num_rows = math.ceil(num_vars / num_cols)
    
    for n, col in enumerate(independent_variables):
        ax = plt.subplot(num_rows, num_cols, n+1)
        ax = sns.countplot(x=col[0], data=dataframe, hue=target_variable, palette=binary_palette)

        ax.set_ylim(0, y_limit)
        ax.set_title(f'{col[1]}', size=20, loc='left', fontweight='bold', pad=15)
        ax.set_xlabel(col[1], size=12)
        ax.set_ylabel('Frequency', size=12)

        label_bars(ax) # function to label the data

    plt.tight_layout()
    plt.show()

def plot_heat_map(dataframe, threshold=0.5):
    corr = dataframe.corr()

    # Create a mask for the upper triangle
    mask_upper = np.triu(np.ones_like(corr, dtype=bool))

    # Apply threshold filter
    mask_threshold = np.abs(corr) < threshold

    # Combine the masks
    final_mask = mask_upper | mask_threshold

    plt.figure(figsize=(20,15))
    sns.heatmap(data=corr, annot=True, mask=final_mask, cmap='coolwarm')

    plt.show()

