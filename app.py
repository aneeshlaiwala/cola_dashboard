import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
import base64

# Load Data
df = pd.read_csv("cola_survey.csv")

# Process Data
age_bins = [18, 24, 34, 44, 54, 100]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55+']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)

features = ['Taste_Rating', 'Price_Rating', 'Packaging_Rating', 'Brand_Reputation_Rating', 'Availability_Rating', 'Sweetness_Rating', 'Fizziness_Rating']
X = df[features]
y = df['NPS_Score']

# Regression Model
reg_model = LinearRegression()
reg_model.fit(X, y)
regression_results = pd.DataFrame({'Feature': features, 'Coefficient': reg_model.coef_})

# Decision Tree Model
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X, y)

df['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X)

# Sidebar Filters
st.sidebar.title("Filters")
age_filter = st.sidebar.multiselect("Select Age Group", df['Age_Group'].unique())
gender_filter = st.sidebar.multiselect("Select Gender", df['Gender'].unique())
income_filter = st.sidebar.multiselect("Select Income Level", df['Income_Level'].unique())

if age_filter:
    df = df[df['Age_Group'].isin(age_filter)]
if gender_filter:
    df = df[df['Gender'].isin(gender_filter)]
if income_filter:
    df = df[df['Income_Level'].isin(income_filter)]

# Main Dashboard
st.title("Cola Survey Interactive Dashboard")

if st.button("Demographic Profile"):
    st.subheader("Gender Distribution")
    gender_fig = px.pie(df, names='Gender', title='Gender Distribution')
    st.plotly_chart(gender_fig)
    
    st.subheader("Age Group Distribution")
    age_fig = px.bar(df, x='Age_Group', title='Age Group Distribution', text_auto=True)
    st.plotly_chart(age_fig)
    
    st.subheader("Income Level Distribution")
    income_fig = px.pie(df, names='Income_Level', title='Income Level Distribution')
    st.plotly_chart(income_fig)

if st.button("Brand Metrics"):
    st.subheader("Most Often Consumed Brand")
    brand_fig = px.bar(df, x='Most_Often_Consumed_Brand', title='Most Often Consumed Brand', text_auto=True)
    st.plotly_chart(brand_fig)
    
    st.subheader("Consumption Frequency")
    freq_fig = px.bar(df, x='Frequency_of_Consumption', title='Consumption Frequency', text_auto=True)
    st.plotly_chart(freq_fig)

if st.button("Basic Attribute Scores"):
    st.subheader("Mean Scores of Attributes")
    mean_scores = df[features].mean().reset_index()
    mean_scores.columns = ['Attribute', 'Mean Score']
    attr_fig = px.bar(mean_scores, x='Attribute', y='Mean Score', title='Attribute Ratings')
    st.plotly_chart(attr_fig)
    
    st.subheader("NPS by Gender")
    nps_gender = px.box(df, x='Gender', y='NPS_Score', title='NPS Score by Gender')
    st.plotly_chart(nps_gender)
    
    st.subheader("NPS by Age Group")
    nps_age = px.box(df, x='Age_Group', y='NPS_Score', title='NPS Score by Age Group')
    st.plotly_chart(nps_age)

if st.button("Regression Analysis"):
    st.subheader("Regression Analysis")
    st.write(regression_results)

if st.button("Cluster Analysis"):
    st.subheader("Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts(normalize=True) * 100
    cluster_fig = px.pie(cluster_counts, names=cluster_counts.index, title='Cluster Distribution')
    st.plotly_chart(cluster_fig)

if st.button("Download Raw Data"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
