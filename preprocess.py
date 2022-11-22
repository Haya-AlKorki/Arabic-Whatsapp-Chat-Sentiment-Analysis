import streamlit as st
import re

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[AaPp][Mm]'
    dates = re.findall(pattern, data)
    messages = re.split(pattern, data)[1:]
    df = pd.DataFrame({'message': messages, 'Date': dates})
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y, %H:%M:%S %p')

    df['message'] = df['message'].str.replace(']', '')
    df['message'] = df['message'].str.replace('[', '')

    users = []
    messages = []
    for message in df['message']:
        entry = re.split(' ([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['User'] = users
    df['Message'] = messages
    df.drop(columns=['message'], inplace=True)

    # Extract date
    df['only_date'] = df['Date'].dt.date

    # Extract year
    df['year'] = df['Date'].dt.year

    # Extract month
    df['month_num'] = df['Date'].dt.month

    # Extract month name
    df['month'] = df['Date'].dt.month_name()

    # Extract day
    df['day'] = df['Date'].dt.day

    # Extract day name
    df['day_name'] = df['Date'].dt.day_name()

    # Extract hour
    df['hour'] = df['Date'].dt.hour

    # Extract minute
    df['minute'] = df['Date'].dt.minute

    # Remove entries having user as group_notification
    df = df[df['User'] != 'group_notification']
    return df