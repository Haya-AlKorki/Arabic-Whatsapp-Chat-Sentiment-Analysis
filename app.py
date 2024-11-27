import streamlit as st
import re
import preprocess,stats, load
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.ticker as ticker
from collections import Counter
import matplotlib.pyplot as plt
import pickle
import plotly.express as px

# App title
st.sidebar.title("Arabic Whatsapp Chat  Sentiment Analyzer")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
# nltk.download('vader_lexicon')

# File upload button
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Main heading
st.markdown("<h1 style='text-align: center; color: limegreen;'>Whatsapp Chat  Sentiment Analyzer</h1>", unsafe_allow_html=True)

if uploaded_file is not None:

    # Getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    d = bytes_data.decode("utf-8")

    # Perform preprocessing
    data = preprocess.preprocess(d)
    data = data.apply(stats.nd, axis=1)

    data = data[data['Normalized_Text'] != ''].reset_index(drop=True)
    data = data.drop_duplicates('Normalized_Text').reset_index(drop=True)

    X = data.dropna()['Normalized_Text'].values
    pred = stats.predict_multi_level(X, load.neutral_vectorizer, load.neutral_lr_model, load.pn_vectorizer, load.lr_model)
    data['Sentiment'] = pred


    data['value'] = data.apply(lambda row: stats.sentiment(row), axis=1)

    st.dataframe(data)

    # User names list
    user_list = data['User'].iloc[1:].unique().tolist()
    # Sorting
    user_list.sort()

    # Insert "Overall" at index 0
    user_list.insert(0, "Overall")

    # Selectbox
    selected_user = st.sidebar.selectbox("Choose a user", user_list)

    if st.sidebar.button("Show Analysis"):


        st.subheader(f'Statistics For: {selected_user}')
        u = len(data['User'].iloc[1:].unique().tolist())
        st.header("Total members:")
        st.title(u)

        col1, col2, col3 = st.columns(3)
        start_dt = str(data['only_date'].iloc[0])[:10]
        last_dt = str(data['only_date'].iloc[-1])[:10]
        with col1:
            st.header("Chat from:")
            st.title(start_dt)
        with col2:
            st.header("Chat to:")
            st.title(last_dt)

        total_messages, total_words = stats.fun_stats(selected_user, data)

        col1, col2 = st.columns(2)
        with col1:
            st.header("Total Messages")
            st.title(total_messages)
        with col2:
            st.header("Total Words")
            st.title(total_words)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = stats.monthly_timeline(selected_user, data,1)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['Normalized_Text'], color='mediumseagreen')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = stats.daily_timeline(selected_user, data,1)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['Normalized_Text'], color='lightgreen')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = stats.week_activity_map(selected_user, data,1)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='lightgreen')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")

            # Ensure the 'Date' column is in datetime format
            data['Date'] = pd.to_datetime(data['Date'],
                                          errors='coerce')  # Replace 'Date' with the actual column name if different

            # Extract month from the datetime
            data['Month'] = data['Date'].dt.month

            # Use your stats function to aggregate data by month (make sure it's implemented in your stats.py)
            busy_month = stats.month_activity_map(selected_user, data, 1)

            # Plot the most busy month
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='lightblue')  # You can change the color if desired
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        ###############################################################################
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x, new_df = stats.most_busy_users(data)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='seagreen')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # pie chart of user activity percentage
        st.markdown("<h3 style='text-align: center; color: black;'>Users Activity in Percentage</h3>", unsafe_allow_html=True)
        user_count = data['User'].iloc[1:].value_counts().reset_index()
        user_count.columns = ['member', 'message']
        fig = px.pie(user_count, names='member', values='message', hole=0.5, color_discrete_sequence=px.colors.sequential.Darkmint_r)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_x=0.5, font={'family': 'Arial', 'size': 16}, xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig, use_container_width=True)


        st.subheader('Word cloud')
        st.write('Top 50 words in all the chat represented as word cloud')
        fig,ax = plt.subplots()
        ax = stats.get_word_cloud(data)
        st.pyplot(fig)

        st.subheader('Word cloud positive')
        st.write('Top 50 potisive words in all the chat represented as word cloud')
        fig,ax = plt.subplots()
        ax = stats.get_word_cloud_positive(data)
        st.pyplot(fig)

        st.subheader('Word cloud neutral')
        st.write('Top 50 neutral words in all the chat represented as word cloud')
        fig,ax = plt.subplots()
        ax = stats.get_word_cloud_neutral(data)
        st.pyplot(fig)

        st.subheader('Word cloud negative')
        st.write('Top 50 words negative in all the chat represented as word cloud')
        fig,ax = plt.subplots()
        ax = stats.get_word_cloud_negative(data)
        st.pyplot(fig)

        # Percentage contributed
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                x = stats.percentage(data, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                y = stats.percentage(data, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                z = stats.percentage(data, -1)

                # Displaying
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            # Getting names per sentiment
            x = data['User'][data['value'] == 1].value_counts().head(10)
            y = data['User'][data['value'] == -1].value_counts().head(10)
            z = data['User'][data['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive <br> Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='darkcyan')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral <br> Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='lightseagreen')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='turquoise')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)


        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)

            busy_month = stats.month_activity_map(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='darkcyan')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_month = stats.month_activity_map(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='lightseagreen')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_month = stats.month_activity_map(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='turquoise')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",
                        unsafe_allow_html=True)

            busy_day = stats.week_activity_map(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='darkcyan')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_day = stats.week_activity_map(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='lightseagreen')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_day = stats.week_activity_map(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='turquoise')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly activity map
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     try:
        #         st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",
        #                     unsafe_allow_html=True)
        #
        #         user_heatmap = stats.activity_heatmap(selected_user, data, 1)
        #
        #         fig, ax = plt.subplots()
        #         ax = sns.heatmap(user_heatmap)
        #         st.pyplot(fig)
        #     except:
        #         st.image('error.webp')
        # with col2:
        #     try:
        #         st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",
        #                     unsafe_allow_html=True)
        #
        #         user_heatmap = stats.activity_heatmap(selected_user, data, 0)
        #
        #         fig, ax = plt.subplots()
        #         ax = sns.heatmap(user_heatmap)
        #         st.pyplot(fig)
        #     except:
        #         st.image('error.webp')
        # with col3:
        #     try:
        #         st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",
        #                     unsafe_allow_html=True)
        #
        #         user_heatmap = stats.activity_heatmap(selected_user, data, -1)
        #
        #         fig, ax = plt.subplots()
        #         ax = sns.heatmap(user_heatmap)
        #         st.pyplot(fig)
        #     except:
        #         st.image('error.webp')

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = stats.daily_timeline(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['Normalized_Text'], color='darkcyan')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = stats.daily_timeline(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['Normalized_Text'], color='lightseagreen')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = stats.daily_timeline(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['Normalized_Text'], color='turquoise')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            timeline = stats.monthly_timeline(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['Normalized_Text'], color='darkcyan')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            timeline = stats.monthly_timeline(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['Normalized_Text'], color='lightseagreen')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            timeline = stats.monthly_timeline(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['Normalized_Text'], color='turquoise')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        # Most common positive words
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     try:
        #         # Data frame of most common positive words.
        #         most_common_df = stats.most_common_words(selected_user, data, 1)
        #
        #         # heading
        #         st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>", unsafe_allow_html=True)
        #         fig, ax = plt.subplots()
        #         ax.barh(most_common_df[0], most_common_df[1], color='blue')
        #         plt.xticks(rotation='vertical')
        #         st.pyplot(fig)
        #     except:
        #         # Disply error image
        #         st.image('error.webp')
        # with col2:
        #     try:
        #         # Data frame of most common neutral words.
        #         most_common_df = stats.most_common_words(selected_user, data, 0)
        #
        #         # heading
        #         st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>", unsafe_allow_html=True)
        #         fig, ax = plt.subplots()
        #         ax.barh(most_common_df[0], most_common_df[1], color='orange')
        #         plt.xticks(rotation='vertical')
        #         st.pyplot(fig)
        #     except:
        #         # Disply error image
        #         st.image('error.webp')
        # with col3:
        #     try:
        #         # Data frame of most common negative words.
        #         most_common_df = stats.most_common_words(selected_user, data, -1)
        #
        #         # heading
        #         st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>", unsafe_allow_html=True)
        #         fig, ax = plt.subplots()
        #         ax.barh(most_common_df[0], most_common_df[1], color='grey')
        #         plt.xticks(rotation='vertical')
        #         st.pyplot(fig)
        #     except:
        #         # Disply error image
        #         st.image('error.webp')

        # Most common positive words
