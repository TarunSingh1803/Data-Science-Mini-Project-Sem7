### Prerequisites
Make sure you have the following libraries installed:
```bash
pip install streamlit pandas matplotlib seaborn scikit-learn plotly
```

### Streamlit Application Code (`streamlit.py`)
```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('Cleaned_categories.csv')
    return data

data = load_data()

# Title of the app
st.title("Cleaned Categories Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Analysis", [
    "Overview",
    "Top 15 Dates",
    "Top 15 Countries",
    "Top 15 Sub-regions",
    "Top 15 Commodities",
    "Top 15 Categories",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Multivariate Analysis",
    "Geographical Analysis",
    "Time Series Analysis",
    "Top Commodities Analysis",
    "Clustering Analysis",
    "Machine Learning Models"
])

# Overview
if options == "Overview":
    st.write("This dashboard provides insights into the Cleaned Categories dataset.")
    st.write(data.head())

# Top 15 Dates
if options == "Top 15 Dates":
    top_dates = data['date'].value_counts().head(15)
    st.bar_chart(top_dates)

# Top 15 Countries
if options == "Top 15 Countries":
    top_countries = data['country_name'].value_counts().head(15)
    st.bar_chart(top_countries)

# Top 15 Sub-regions
if options == "Top 15 Sub-regions":
    top_subregions = data['sub_region'].value_counts().head(15)
    st.bar_chart(top_subregions)

# Top 15 Commodities
if options == "Top 15 Commodities":
    top_commodities = data['commodity'].value_counts().head(15)
    st.bar_chart(top_commodities)

# Top 15 Categories
if options == "Top 15 Categories":
    top_categories = data['categories'].value_counts().head(15)
    st.bar_chart(top_categories)

# Univariate Analysis
if options == "Univariate Analysis":
    st.write("Univariate Analysis of Value Quantities")
    sns.histplot(data['value_qt'], bins=30)
    st.pyplot()

# Bivariate Analysis
if options == "Bivariate Analysis":
    st.write("Bivariate Analysis: Value Quantity vs Value in Rs")
    sns.scatterplot(x='value_qt', y='value_rs', data=data)
    st.pyplot()

# Multivariate Analysis
if options == "Multivariate Analysis":
    st.write("Multivariate Analysis: Pairplot")
    sns.pairplot(data[['value_qt', 'value_rs', 'country_code']])
    st.pyplot()

# Geographical Analysis
if options == "Geographical Analysis":
    st.write("Geographical Analysis of Value Quantities")
    fig = px.choropleth(data, locations='country_code', locationmode='ISO-3',
                        color='value_qt', hover_name='country_name', 
                        color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig)

# Time Series Analysis
if options == "Time Series Analysis":
    data['date'] = pd.to_datetime(data['date'])
    time_series = data.groupby('date')['value_qt'].sum().reset_index()
    st.line_chart(time_series.set_index('date'))

# Top Commodities Analysis
if options == "Top Commodities Analysis":
    top_commodities = data.groupby('commodity')['value_qt'].sum().nlargest(15)
    st.bar_chart(top_commodities)

# Clustering Analysis
if options == "Clustering Analysis":
    st.write("K-Means Clustering Analysis")
    kmeans_data = data[['value_qt', 'value_rs']]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(kmeans_data)
    data['cluster'] = kmeans.labels_
    sns.scatterplot(x='value_qt', y='value_rs', hue='cluster', data=data)
    st.pyplot()

# Machine Learning Models
if options == "Machine Learning Models":
    st.write("Simple Linear Regression")
    X = data[['value_qt']]
    y = data['value_rs']
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    plt.scatter(X, y)
    plt.plot(X, predictions, color='red')
    st.pyplot()

    st.write("Multiple Linear Regression")
    # Assuming you have more features for multiple regression
    # X_multi = data[['value_qt', 'another_feature']]
    # model_multi = LinearRegression()
    # model_multi.fit(X_multi, y)
    # predictions_multi = model_multi.predict(X_multi)
    # plt.scatter(X_multi['value_qt'], y)
    # plt.plot(X_multi['value_qt'], predictions_multi, color='red')
    # st.pyplot()

    st.write("Logistic Regression")
    # Logistic regression example (requires binary target variable)
    # from sklearn.linear_model import LogisticRegression
    # model_logistic = LogisticRegression()
    # model_logistic.fit(X, y_binary)
    # predictions_logistic = model_logistic.predict(X)
    # st.write(predictions_logistic)

    st.write("K-Means Clustering")
    # Already covered above

# Run the app
if __name__ == "__main__":
    st.run()
```

### Explanation of the Code
1. **Data Loading**: The dataset is loaded using `pandas` and cached for performance.
2. **Navigation**: A sidebar allows users to navigate between different analyses.
3. **Visualizations**: Various analyses are performed, including bar charts for top items, histograms for univariate analysis, scatter plots for bivariate analysis, and geographical visualizations using Plotly.
4. **Machine Learning Models**: Simple linear regression is demonstrated, and placeholders for multiple linear regression and logistic regression are included.

### Running the Application
To run the application, navigate to the directory containing `streamlit.py` and execute:
```bash
streamlit run streamlit.py
```

### Note
This is a basic implementation. You may need to adjust the code based on the actual structure of your dataset and the specific analyses you want to perform. Additionally, ensure that the dataset is in the same directory as the script or provide the correct path to the CSV file.