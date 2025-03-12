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
    return pd.read_csv("cola_survey.csv")

df = load_data()

# Streamlit App Title
st.title("Interactive Cola Consumer Dashboard")

# Sidebar Filters
brand = st.sidebar.selectbox("Select a Brand", df["Brand_Preference"].unique())
gender = st.sidebar.selectbox("Select Gender", df["Gender"].unique())
income = st.sidebar.selectbox("Select Income Level", df["Income_Level"].unique())

# Filter Data
filtered_df = df[(df["Brand_Preference"] == brand) & (df["Gender"] == gender) & (df["Income_Level"] == income)]

# Demographic Profile
if st.button("Demographic Profile"):
    st.subheader("Gender Distribution")
    fig = px.pie(df, names='Gender', title='Gender Distribution')
    st.plotly_chart(fig)
    
    st.subheader("Age Distribution")
    fig = px.histogram(df, x='Age', nbins=10, title='Age Distribution')
    st.plotly_chart(fig)
    
    st.subheader("Income Level Distribution")
    fig = px.pie(df, names='Income_Level', title='Income Level Distribution')
    st.plotly_chart(fig)

# Brand Metrics
if st.button("Brand Metrics"):
    st.subheader("Most Often Used Brand")
    fig = px.bar(df, x='Most_Often_Consumed_Brand', title='Most Often Used Brand')
    st.plotly_chart(fig)
    
    st.subheader("Occasions of Buying")
    fig = px.bar(df, x='Occasions_of_Buying', title='Occasions of Buying')
    st.plotly_chart(fig)
    
    st.subheader("Frequency of Consumption")
    fig = px.bar(df, x='Frequency_of_Consumption', title='Frequency of Consumption')
    st.plotly_chart(fig)

# Basic Attribute Scores
if st.button("Basic Attribute Scores"):
    attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']
    avg_scores = df[attributes].mean()
    st.bar_chart(avg_scores)
    st.subheader("NPS Score Distribution")
    fig = px.histogram(df, x='NPS_Score', nbins=10, title='NPS Score Distribution')
    st.plotly_chart(fig)

# Regression Analysis
if st.button("Regression Analysis"):
    st.subheader("Regression Analysis")
    X = df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']]
    y = df['NPS_Score']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    st.text(model.summary())

# Decision Tree Analysis
if st.button("Answer Decision Tree"):
    st.subheader("Decision Tree Analysis")
    X_tree = X.copy()
    y_tree = df['NPS_Score'].apply(lambda x: 1 if x >= 9 else 0)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_tree, y_tree)
    fig, ax = plt.subplots(figsize=(12, 6))
    tree.plot_tree(clf, feature_names=X_tree.columns, class_names=['Detractor/Passive', 'Promoter'], filled=True, fontsize=8, ax=ax)
    st.pyplot(fig)

# Cluster Analysis
if st.button("Cluster Analysis"):
    st.subheader("Customer Segmentation")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    fig = px.scatter(df, x='Taste_Rating', y='Fizziness_Rating', color=df['Cluster'].astype(str), title='Cluster Distribution')
    st.plotly_chart(fig)
    
    st.write("### Cluster Descriptions:")
    st.write("1. **Fizz-Lovers** - Customers who prefer high carbonation levels.")
    st.write("2. **Brand-Conscious Consumers** - Customers who prefer strong branding and reputation.")
    st.write("3. **Budget-Friendly Drinkers** - Customers who prioritize price and availability over taste.")
