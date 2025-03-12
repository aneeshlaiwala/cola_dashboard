import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import statsmodels.api as sm

# Load dataset from GitHub or local repo
@st.cache_data
def load_data():
    return pd.read_csv("cola_survey.csv")

df = load_data()

# Streamlit App Title
st.title("Interactive Cola Consumer Dashboard")

# Sidebar Filters
brand = st.sidebar.selectbox("Select a Brand", df["Brand_Preference"].unique())

# Buttons for different analyses
if st.button("Show Basic Stats"):
    st.subheader("Basic Statistics")
    st.write(df.describe())

if st.button("Show Brand Ratings"):
    st.subheader(f"Average Ratings for {brand}")
    avg_ratings = df[df["Brand_Preference"] == brand][['Taste_Rating', 'Price_Rating', 'Fizziness_Rating']].mean()
    st.bar_chart(avg_ratings)

if st.button("Show NPS Score Distribution"):
    st.subheader("NPS Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[df["Brand_Preference"] == brand]["NPS_Score"], bins=10, kde=True, ax=ax)
    ax.set_title(f"NPS Score Distribution for {brand}")
    st.pyplot(fig)

if st.button("Run Clustering Analysis"):
    st.subheader("K-Means Clustering")
    num_clusters = st.slider("Select Number of Clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[['Taste_Rating', 'Price_Rating', 'Fizziness_Rating']])
    fig = px.scatter(df, x="Taste_Rating", y="Fizziness_Rating", color=df["Cluster"].astype(str))
    st.plotly_chart(fig)

if st.button("Run Regression Analysis"):
    st.subheader("Regression Analysis")
    X = df[['Taste_Rating', 'Price_Rating', 'Fizziness_Rating']]
    y = df['NPS_Score']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    st.text(model.summary())

if st.button("Show Data Table"):
    st.subheader("Filtered Data Table")
    st.dataframe(df[df["Brand_Preference"] == brand])
