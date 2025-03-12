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
brand = st.sidebar.selectbox("Select a Brand", [None] + list(df["Brand_Preference"].unique()))
gender = st.sidebar.selectbox("Select Gender", [None] + list(df["Gender"].unique()))
income = st.sidebar.selectbox("Select Income Level", [None] + list(df["Income_Level"].unique()))

# Filter Data
filtered_df = df.copy()
if brand:
    filtered_df = filtered_df[filtered_df["Brand_Preference"] == brand]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"] == gender]
if income:
    filtered_df = filtered_df[filtered_df["Income_Level"] == income]

# Demographic Profile
if st.button("Demographic Profile"):
    st.subheader("Gender Distribution")
    fig = px.pie(df, names='Gender', title='Gender Distribution')
    st.plotly_chart(fig)
    
    st.subheader("Age Distribution (Grouped)")
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-24', '25-34', '35-44', '45-54', '55-64'])
    fig = px.histogram(df, x='Age_Group', title='Age Group Distribution')
    st.plotly_chart(fig)
    
    st.subheader("Income Level Distribution")
    fig = px.pie(df, names='Income_Level', title='Income Level Distribution')
    st.plotly_chart(fig)

# Brand Metrics
if st.button("Brand Metrics"):
    st.subheader("Most Often Used Brand (Percentage)")
    brand_counts = df['Most_Often_Consumed_Brand'].value_counts(normalize=True) * 100
    fig = px.bar(x=brand_counts.index, y=brand_counts.values, text=brand_counts.values, title='Most Often Used Brand')
    st.plotly_chart(fig)
    
    st.subheader("Occasions of Buying (Percentage)")
    occasions_counts = df['Occasions_of_Buying'].value_counts(normalize=True) * 100
    fig = px.bar(x=occasions_counts.index, y=occasions_counts.values, text=occasions_counts.values, title='Occasions of Buying')
    st.plotly_chart(fig)
    
    st.subheader("Frequency of Consumption (Percentage)")
    freq_counts = df['Frequency_of_Consumption'].value_counts(normalize=True) * 100
    fig = px.bar(x=freq_counts.index, y=freq_counts.values, text=freq_counts.values, title='Frequency of Consumption')
    st.plotly_chart(fig)

# Basic Attribute Scores
if st.button("Basic Attribute Scores"):
    attributes = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']
    avg_scores = df[attributes].mean()
    st.bar_chart(avg_scores)
    
    st.subheader("NPS Score Distribution by Age")
    nps_avg_by_age = df.groupby('Age_Group')['NPS_Score'].mean()
    st.bar_chart(nps_avg_by_age)

# Regression Analysis
if st.button("Regression Analysis"):
    st.subheader("Regression Analysis")
    X = df[['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']]
    y = df['NPS_Score']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    st.text(model.summary())
    st.write("### Summary of Findings:")
    st.write("- The most significant factors impacting NPS are ...")

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
    st.write("### Summary Conclusion:")
    st.write("- The key factors influencing promoters vs detractors are ...")

# Cluster Analysis
if st.button("Cluster Analysis"):
    st.subheader("Customer Segmentation")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    cluster_counts = df['Cluster'].value_counts(normalize=True) * 100
    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, text=cluster_counts.values, title='Cluster Distribution (%)')
    st.plotly_chart(fig)
    
    st.write("### Cluster Descriptions & Conclusions:")
    st.write("1. **Fizz-Lovers** - Customers who prefer high carbonation levels. Conclusion: ...")
    st.write("2. **Brand-Conscious Consumers** - Customers who prefer strong branding and reputation. Conclusion: ...")
    st.write("3. **Budget-Friendly Drinkers** - Customers who prioritize price and availability over taste. Conclusion: ...")

# Executive Summary
if st.button("Executive Summary"):
    st.write("### Key Findings:")
    st.write("- The demographic distribution indicates ...")
    st.write("- The most used brand is ...")
    st.write("- The key factors impacting NPS are ...")
    st.write("- Clustering revealed three main segments ...")

# Download Data
if st.button("Download Full Dataset"):
    st.subheader("Download the Entire Dataset")
    st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name="cola_survey_data.csv", mime="text/csv")
