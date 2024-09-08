import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the default style for matplotlib
plt.style.use('fivethirtyeight')  # Using 'default' instead of 'seaborn'

# Set page config
st.set_page_config(page_title="VeBetterDAO: X-apps User Insights", layout="centered")

# Load the dataset
df = pd.read_csv('dataframe.csv')

# Convert Timestamp and FirstSeen to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['FirstSeen'] = pd.to_datetime(df['FirstSeen'])
df['DayOfWeek'] = pd.Categorical(df['Timestamp'].dt.day_name(), categories=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
df['HourOfDay'] = df['Timestamp'].dt.hour
df['DaysSinceFirst'] = (df['Timestamp'] - df['FirstSeen']).dt.total_seconds() / (60 * 60 * 24)  # in days
df['Weekend'] = df['Timestamp'].dt.dayofweek >= 5  # Weekend if True

st.title("VeBetterDAO: X-apps User Insights Dashboard")

st.write("Note: This data primarily focuses on apps utilizing the Rewards Pool for B3TR distribution.")

# Set the style for all plots

# Create a function to display the visualizations
def create_visualizations():
    
    st.subheader("Total Number of Actions per Day")
    df_total_actions = df.groupby(df['Timestamp'].dt.date).size()
    fig, ax = plt.subplots(figsize=(8, 5))
    df_total_actions.plot(ax=ax, color='blue')
    ax.set_title("Total Number of Actions per Day")
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Actions")
    st.pyplot(fig)
    
    # NEW: Cumulative Total Actions Over Time
    st.subheader("Cumulative Total Actions Over Time")
    df_cumulative_actions = df_total_actions.cumsum()
    fig, ax = plt.subplots(figsize=(8, 5))
    df_cumulative_actions.plot(ax=ax, color='green')
    ax.set_title("Cumulative Total Actions Over Time")
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Actions")
    st.pyplot(fig)
    
    # Chart 1: App Usage Over Time
    st.subheader("App Usage Over Time")
    df_grouped_time = df.groupby([df['Timestamp'].dt.date, 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for app in df_grouped_time.columns:
        if app == 'GreenAmbassador':
            ax.plot(df_grouped_time.index, df_grouped_time[app], linestyle='--', marker='o', label=app)
        else:
            ax.plot(df_grouped_time.index, df_grouped_time[app], label=app)
    ax.set_title("App Usage Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart shows the daily usage of each app over time, allowing us to see trends and patterns in app popularity. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 2: User Engagement Over Time
    st.subheader("User Engagement Over Time")
    
    # Calculate cumulative unique users for each app
    df_cumulative_users = df.groupby(['App', df['Timestamp'].dt.date])['Receiver'].nunique().groupby(level=0).cumsum().reset_index()
    df_cumulative_users.columns = ['App', 'Date', 'Cumulative_Users']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    for app in df_cumulative_users['App'].unique():
        app_data = df_cumulative_users[df_cumulative_users['App'] == app]
        if app == 'GreenAmbassador':
            ax.plot(app_data['Date'], app_data['Cumulative_Users'], label=app, linewidth=2, linestyle='--', marker='o')
        else:
            ax.plot(app_data['Date'], app_data['Cumulative_Users'], label=app, linewidth=2)
    
    ax.set_title("Cumulative Unique Users Over Time by App", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Unique Users", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("This graph shows the cumulative growth of unique users for each app over time. It helps visualize user acquisition and retention trends across different apps. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 3: Total Reward Distributed per App
    st.subheader("Total Reward Distributed per App")
    df_grouped_reward = df.groupby('App')['Reward'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_grouped_reward.plot(kind='bar', ax=ax, color=plt.cm.Set3(np.arange(len(df_grouped_reward))))
    ax.set_title("Total Reward Distributed per App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This chart displays the total rewards distributed by each app, giving insight into which apps are most generous or popular in terms of rewards.")
    
    # Chart 4: Unique Receivers/Wallets per App
    st.subheader("Unique Receivers/Wallets per App")
    df_unique_receivers = df.groupby('App')['Receiver'].nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_unique_receivers.plot(kind='bar', ax=ax, color=plt.cm.Set2(np.arange(len(df_unique_receivers))))
    ax.set_title("Unique Receivers/Wallets per App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Unique Receivers/Wallets", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This graph shows the number of unique users (wallets) for each app, indicating the size of each app's user base.")
    
    # Chart 5: Average Reward per User per App
    st.subheader("Average Reward per User per App")
    df_avg_reward = df.groupby(['App', 'Receiver'])['Reward'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='App', y='Reward', data=df_avg_reward, ax=ax, palette='Set1')
    ax.set_title("Average Reward per User per App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot visualizes the distribution of average rewards per user for each app, helping to identify which apps tend to give higher or more consistent rewards.")
    
    # Chart 6: Daily Actions per App (Stacked Area)
    st.subheader("Daily Actions per App (Stacked Area)")
    df_daily_actions = df.groupby([df['Timestamp'].dt.date, 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_daily_actions.plot(kind='area', stacked=True, ax=ax, cmap='viridis')
    ax.set_title("Daily Actions per App", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This stacked area chart shows the daily number of actions for each app, allowing us to see how app usage evolves over time and in relation to each other.")
    
    # Chart 7: App Usage by Day of the Week (Heatmap)
    st.subheader("App Usage by Day of the Week (Heatmap)")
    df_heatmap = df.groupby(['DayOfWeek', 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_heatmap.T, cmap='YlGnBu', annot=True, fmt='g', ax=ax, cbar_kws={'label': 'Number of Actions'})
    ax.set_title("App Usage by Day of the Week", fontsize=16)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("App", fontsize=12)
    st.pyplot(fig)
    st.write("This heatmap visualizes app usage patterns across different days of the week, helping to identify peak usage days for each app. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 8: Total Rewards Distributed Over Time
    st.subheader("Total Rewards Distributed Over Time")
    df_rewards_time = df.groupby([df['Timestamp'].dt.date, 'App'])['Reward'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for app in df_rewards_time.columns:
        if app == 'GreenAmbassador':
            ax.plot(df_rewards_time.index, df_rewards_time[app], linestyle='--', marker='o', label=app)
        else:
            ax.plot(df_rewards_time.index, df_rewards_time[app], label=app)
    ax.set_title("Total Rewards Distributed Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart shows how the total rewards distributed by each app change over time, which can indicate changes in app popularity or reward strategies. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 9: First-time Users Over Time
    st.subheader("First-time Users Over Time")
    df_first_time_users = df.groupby([df['FirstSeen'].dt.date, 'App'])['Receiver'].nunique().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for app in df_first_time_users.columns:
        if app == 'GreenAmbassador':
            ax.plot(df_first_time_users.index, df_first_time_users[app], linestyle='--', marker='o', label=app)
        else:
            ax.plot(df_first_time_users.index, df_first_time_users[app], label=app)
    ax.set_title("First-time Users Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of First-Time Users", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This graph shows the number of new users joining each app over time, helping to visualize user acquisition trends. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 10: Reward Distribution by Day of the Week
    st.subheader("Reward Distribution by Day of the Week")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='DayOfWeek', y='Reward', data=df, palette="Set2", ax=ax)
    ax.set_title("Reward Distribution by Day of the Week", fontsize=16)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot shows how rewards are distributed across different days of the week, which can reveal patterns in reward allocation. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 11: Cumulative Rewards per App Over Time
    st.subheader("Cumulative Rewards per App Over Time")
    df_cumulative_rewards = df.groupby([df['Timestamp'].dt.date, 'App'])['Reward'].sum().groupby('App').cumsum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for app in df_cumulative_rewards.columns:
        if app == 'GreenAmbassador':
            ax.plot(df_cumulative_rewards.index, df_cumulative_rewards[app], linestyle='--', marker='o', label=app)
        else:
            ax.plot(df_cumulative_rewards.index, df_cumulative_rewards[app], label=app)
    ax.set_title("Cumulative Rewards per App Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart displays the cumulative rewards distributed by each app over time, showing the total value generated by each app. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 12: Actions by Hour of the Day
    st.subheader("Actions by Hour of the Day")
    df_actions_hour = df.groupby(['HourOfDay', 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_actions_hour.plot(ax=ax, linewidth=2)
    ax.set_title("Actions by Hour of the Day", fontsize=16)
    ax.set_xlabel("Hour", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This graph shows the number of actions taken in each app throughout the day, revealing peak usage hours for each app.")
    
    # Chart 13: Unique Receivers Growth Over Time
    st.subheader("Unique Receivers Growth Over Time")
    df_unique_receivers_growth = df.groupby([df['Timestamp'].dt.date, 'App'])['Receiver'].nunique().groupby('App').cumsum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for app in df_unique_receivers_growth.columns:
        if app == 'GreenAmbassador':
            ax.plot(df_unique_receivers_growth.index, df_unique_receivers_growth[app], linestyle='--', marker='o', label=app)
        else:
            ax.plot(df_unique_receivers_growth.index, df_unique_receivers_growth[app], label=app)
    ax.set_title("Unique Receivers Growth Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Unique Receivers", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart shows the cumulative growth of unique users for each app over time, helping to visualize user acquisition and retention trends.")
    
    # Chart 14: Actions per App per Receiver
    st.subheader("Actions per App per Receiver")
    df_actions_per_user = df.groupby(['App', 'Receiver']).size().reset_index(name='Actions')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='App', y='Actions', data=df_actions_per_user, palette="muted", ax=ax)
    ax.set_title("Actions per App per Receiver", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot shows the distribution of the number of actions taken by users in each app, indicating user engagement levels across different apps.")
    
    # Chart 15: Time Between First Action and Current Action (Days)
    st.subheader("Time Between First Action and Current Action (Days)")
    df['TimeSinceFirst'] = (df['Timestamp'] - df['FirstSeen']).dt.total_seconds() / (60 * 60 * 24)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='App', y='TimeSinceFirst', data=df, palette="coolwarm", ax=ax)
    ax.set_title("Time Between First Action and Current Action (Days)", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Days Since First Action", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot visualizes the distribution of time between a user's first action and their subsequent actions for each app, indicating user retention and long-term engagement.")
    
    # Chart 16: Heatmap of Total Rewards by App and Day of the Week
    st.subheader("Heatmap of Total Rewards by App and Day of the Week")
    df_heatmap_rewards = df.groupby([df['DayOfWeek'], 'App'])['Reward'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_heatmap_rewards.T, cmap='viridis', annot=True, fmt='.0f', ax=ax, cbar_kws={'label': 'Total Reward'})
    ax.set_title("Total Rewards by App and Day of the Week", fontsize=16)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("App", fontsize=12)
    st.pyplot(fig)
    
    st.subheader("Distribution of Days Since First Interaction")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['DaysSinceFirst'], kde=True, bins=30, color='purple', ax=ax)
    ax.set_title("Distribution of Days Since First Interaction", fontsize=16)
    ax.set_xlabel("Days Since First Interaction", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig)
    
    st.subheader("App Usage: Weekdays vs Weekends")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Weekend', hue='App', data=df, palette='Set3', ax=ax)
    ax.set_title("App Usage: Weekdays vs Weekends", fontsize=16)
    ax.set_xlabel("Is Weekend", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    
    st.subheader("Receiver Retention by App")
    df_retention = df.groupby(['App', 'Receiver']).size().groupby('App').mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_retention.plot(kind='bar', ax=ax, color=plt.cm.Set3(np.arange(len(df_retention))))
    ax.set_title("Average Retention (Actions per Receiver) by App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Average Actions per Receiver", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    st.subheader("Total Reward vs. Unique Receivers per App")
    df_total_reward = df.groupby('App')['Reward'].sum().reset_index()
    df_unique_receivers = df.groupby('App')['Receiver'].nunique().reset_index()
    df_reward_vs_receivers = pd.merge(df_total_reward, df_unique_receivers, on='App')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Receiver', y='Reward', data=df_reward_vs_receivers, hue='App', s=100, ax=ax)
    ax.set_title("Total Reward vs. Unique Receivers per App", fontsize=16)
    ax.set_xlabel("Unique Receivers", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    
   

# Generate and display the visualizations
create_visualizations()