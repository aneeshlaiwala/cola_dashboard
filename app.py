import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import statsmodels.api as sm

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cola_survey.csv")
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-24', '25-34', '35-44', '45-54', '55-64'], ordered=True)
    return df

df = load_data()

# Streamlit App Title
st.title("Interactive Cola Consumer Dashboard")

# Cluster Analysis (Precompute and Append to Data)
X_cluster = df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)
df['Cluster_Name'] = df['Cluster'].map({0: 'Fizz-Lovers', 1: 'Brand-Conscious Consumers', 2: 'Budget-Friendly Drinkers'})

# Sidebar Filters
brand = st.sidebar.selectbox("Select a Brand", [None] + list(df["Brand_Preference"].unique()))
gender = st.sidebar.selectbox("Select Gender", [None] + list(df["Gender"].unique()))
income = st.sidebar.selectbox("Select Income Level", [None] + list(df["Income_Level"].unique()))
cluster = st.sidebar.selectbox("Select Cluster", [None] + list(df["Cluster_Name"].unique()))

# Filter Data
filtered_df = df.copy()
if brand:
    filtered_df = filtered_df[filtered_df["Brand_Preference"] == brand]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"] == gender]
if income:
    filtered_df = filtered_df[filtered_df["Income_Level"] == income]
if cluster:
    filtered_df = filtered_df[filtered_df["Cluster_Name"] == cluster]

# Brand Metrics
if st.button("Brand Metrics"):
    st.subheader("Most Often Used Brand (Percentage)")
    brand_counts = filtered_df['Most_Often_Consumed_Brand'].value_counts(normalize=True) * 100
    fig = px.bar(x=brand_counts.index, y=brand_counts.values.round(1), text=brand_counts.values.round(1), title='Most Often Used Brand')
    st.plotly_chart(fig)
    
    st.subheader("Occasions of Buying (Percentage)")
    occasions_counts = filtered_df['Occasions_of_Buying'].value_counts(normalize=True) * 100
    fig = px.bar(x=occasions_counts.index, y=occasions_counts.values.round(1), text=occasions_counts.values.round(1), title='Occasions of Buying')
    st.plotly_chart(fig)
    
    st.subheader("Frequency of Consumption (Percentage)")
    freq_counts = filtered_df['Frequency_of_Consumption'].value_counts(normalize=True) * 100
    fig = px.bar(x=freq_counts.index, y=freq_counts.values.round(1), text=freq_counts.values.round(1), title='Frequency of Consumption')
    st.plotly_chart(fig)

# Cluster Analysis
if st.button("Cluster Analysis"):
    st.subheader("Customer Segmentation")
    cluster_counts = filtered_df['Cluster_Name'].value_counts(normalize=True) * 100
    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values.round(1), text=cluster_counts.values.round(1), title='Cluster Distribution (%)')
    st.plotly_chart(fig)

# View and Download Full Dataset
if st.button("View & Download Full Dataset"):
    st.subheader("Full Dataset")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="cola_survey_data.csv", mime="text/csv")
