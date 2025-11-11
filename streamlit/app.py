import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier




from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Country Insights Function
def country_insights(country_name, df):
    """
    Produce quick EDA + model-based insights for a chosen country.
    Expects 'df' to be the cleaned dataframe used in the notebook.
    """
    # Basic check
    df_country = df[df['country_name'].astype(str).str.strip().eq(country_name)]
    if df_country.empty:
        st.error(f"No records found for country: {country_name}")
        return None

    # Summary statistics
    st.subheader(f"📊 Country Analysis: {country_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df_country):,}")
    with col2:
        total_value_usd = df_country['value_dl'].sum()
        st.metric("Total Export Value (USD)", f"${total_value_usd:,.2f}")
    with col3:
        mean_value = df_country['value_dl'].mean()
        st.metric("Mean Value per Consignment", f"${mean_value:,.2f}")
    with col4:
        median_value = df_country['value_dl'].median()
        st.metric("Median Value per Consignment", f"${median_value:,.2f}")

    # Additional metrics if quantity data is available
    if 'value_qt' in df_country.columns:
        total_qty = df_country['value_qt'].sum()
        st.metric("Total Quantity (QT)", f"{total_qty:,.2f}")

    # Top commodities analysis
    st.subheader("🏆 Top 10 Commodities by Total Value")
    top_commodities_value = df_country.groupby('commodity')['value_dl'].sum().sort_values(ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_commodities_value.plot(kind='barh', ax=ax)
        ax.set_title(f'Top 10 Commodities by Value - {country_name}')
        ax.set_xlabel('Total Value (USD)')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.dataframe(top_commodities_value.reset_index())

    # Top commodities by count
    st.subheader("📦 Top 10 Commodities by Frequency")
    top_commodities_count = df_country['commodity'].value_counts().head(10)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_commodities_count.plot(kind='barh', ax=ax)
        ax.set_title(f'Top 10 Commodities by Count - {country_name}')
        ax.set_xlabel('Number of Records')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.dataframe(top_commodities_count.reset_index())

    # Category distribution
    if 'categories' in df_country.columns:
        st.subheader("📂 Category Distribution")
        category_dist = df_country['categories'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            category_dist.head(10).plot(kind='pie', ax=ax)
            ax.set_title(f'Category Distribution - {country_name}')
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.dataframe(category_dist.reset_index())

    # Time series analysis
    st.subheader("📈 Monthly Export Value Trend")
    try:
        if 'date' in df_country.columns:
            df_country_copy = df_country.copy()
            df_country_copy['date'] = pd.to_datetime(df_country_copy['date'])
            ts_country = df_country_copy.set_index('date')['value_dl'].resample('M').sum().reset_index()
            
            if ts_country['value_dl'].sum() > 0:
                fig = px.line(ts_country, x='date', y='value_dl',
                              title=f"Monthly Export Value (USD) — {country_name}",
                              labels={'value_dl': 'Value (USD)', 'date': 'Month'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly time-series values to plot for this country.")
    except Exception as e:
        st.error(f"Could not create time series plot: {e}")

    # Heatmap: top commodities vs months
    st.subheader("🔥 Commodity Performance Heatmap")
    try:
        if 'date' in df_country.columns:
            df_country_copy = df_country.copy()
            df_country_copy['date'] = pd.to_datetime(df_country_copy['date'])
            pivot = df_country_copy.pivot_table(values='value_dl',
                                           index='commodity',
                                           columns=df_country_copy['date'].dt.to_period('M'),
                                           aggfunc='sum').fillna(0)
            top_comm = pivot.sum(axis=1).sort_values(ascending=False).head(10).index
            
            if len(top_comm) > 0:
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.heatmap(pivot.loc[top_comm].T, cmap='YlGnBu', ax=ax)
                ax.set_title(f"Heatmap: Top commodities (value) over months — {country_name}")
                ax.set_xlabel("Commodity")
                ax.set_ylabel("Month")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not create heatmap: {e}")

    return {
        'country_name': country_name,
        'total_records': len(df_country),
        'total_value_usd': total_value_usd,
        'mean_value': mean_value,
        'median_value': median_value,
        'top_commodities_value': top_commodities_value,
        'top_commodities_count': top_commodities_count,
        'category_distribution': category_dist if 'categories' in df_country.columns else pd.Series(),
        'raw_data': df_country
    }

# Sub-region Insights Function
def subregion_insights(sub_region_name, df):
    """
    Produce aggregated EDA + visualizations for a chosen sub-region.
    - sub_region_name : str, e.g. "Western Europe"
    - df : DataFrame (cleaned)
    """
    # Filter sub-region (case-insensitive)
    mask = df['sub_region'].astype(str).str.strip().str.lower() == str(sub_region_name).strip().lower()
    df_sub = df[mask].copy()
    if df_sub.empty:
        st.error(f"No records found for sub-region: {sub_region_name}")
        return None

    # Basic summary
    total_value = df_sub['value_dl'].sum()
    total_qty = df_sub['value_qt'].sum() if 'value_qt' in df_sub.columns else np.nan
    n_records = len(df_sub)
    n_countries = df_sub['country_name'].nunique()

    st.subheader(f"🌍 Sub-region Analysis: {sub_region_name}")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{n_records:,}")
    with col2:
        st.metric("Number of Countries", f"{n_countries}")
    with col3:
        st.metric("Total Export Value (USD)", f"${total_value:,.2f}")
    with col4:
        mean_value = df_sub['value_dl'].mean()
        st.metric("Mean Value per Consignment", f"${mean_value:,.2f}")

    if not np.isnan(total_qty):
        st.metric("Total Quantity (QT)", f"{total_qty:,.2f}")

    # Top countries in sub-region
    st.subheader("🏆 Top 10 Countries by Total Export Value")
    by_value = df_sub.groupby('country_name')['value_dl'].sum().sort_values(ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=by_value.values, y=by_value.index, orientation='h',
                     title=f'Top 10 Countries by Value - {sub_region_name}',
                     labels={'x': 'Total Value (USD)', 'y': 'Country'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(by_value.reset_index().rename(columns={'country_name': 'Country', 'value_dl': 'Total Value (USD)'}))

    # Top countries by consignment count
    st.subheader("📦 Top 10 Countries by Number of Consignments")
    by_count = df_sub['country_name'].value_counts().head(10)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=by_count.values, y=by_count.index, orientation='h',
                     title=f'Top 10 Countries by Consignments - {sub_region_name}',
                     labels={'x': 'Number of Consignments', 'y': 'Country'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(by_count.reset_index().rename(columns={'country_name': 'Country', 'count': 'Number of Consignments'}))

    # Top commodities analysis
    st.subheader("🏅 Top 10 Commodities by Total Value")
    top_commodities = df_sub.groupby('commodity')['value_dl'].sum().sort_values(ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=top_commodities.values, y=top_commodities.index, orientation='h',
                     title=f'Top 10 Commodities by Value - {sub_region_name}',
                     labels={'x': 'Total Value (USD)', 'y': 'Commodity'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(top_commodities.reset_index().rename(columns={'commodity': 'Commodity', 'value_dl': 'Total Value (USD)'}))

    # Category distribution
    if 'categories' in df_sub.columns:
        st.subheader("📂 Category Distribution")
        category_dist = df_sub['categories'].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=category_dist.values, names=category_dist.index,
                         title=f'Category Distribution - {sub_region_name}')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(category_dist.reset_index().rename(columns={'categories': 'Category', 'count': 'Number of Records'}))

    # Time series analysis
    st.subheader("📈 Monthly Export Value Trend")
    try:
        if 'date' in df_sub.columns:
            df_sub_copy = df_sub.copy()
            df_sub_copy['date'] = pd.to_datetime(df_sub_copy['date'])
            ts_sub = df_sub_copy.set_index('date')['value_dl'].resample('M').sum().reset_index()
            
            if ts_sub['value_dl'].sum() > 0:
                fig = px.line(ts_sub, x='date', y='value_dl',
                              title=f"Monthly Export Value (USD) — {sub_region_name}",
                              labels={'value_dl': 'Value (USD)', 'date': 'Month'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly time-series values to plot for this sub-region.")
    except Exception as e:
        st.error(f"Could not create time series plot: {e}")

    # Choropleth map for countries in sub-region
    st.subheader("🗺️ Geographic Distribution of Export Values")
    try:
        if 'alpha_3_code' in df_sub.columns:
            country_value = df_sub.groupby(['country_name', 'alpha_3_code'])['value_dl'].sum().reset_index()
            if country_value['alpha_3_code'].notna().any():
                fig_map = px.choropleth(country_value,
                                        locations="alpha_3_code",
                                        color="value_dl",
                                        hover_name="country_name",
                                        hover_data={'value_dl': ':,.2f'},
                                        color_continuous_scale=px.colors.sequential.Plasma,
                                        title=f"Total Export Value (USD) by Country — {sub_region_name}")
                fig_map.update_layout(height=600)
                st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Geographic visualization requires country codes (alpha_3_code column).")
    except Exception as e:
        st.warning(f"Could not create geographic map: {e}")

    # Country summary table
    st.subheader("📋 Country Summary Table")
    country_summary = df_sub.groupby('country_name').agg({
        'value_dl': ['count', 'sum', 'mean', 'median'],
        'commodity': 'nunique'
    }).round(2)
    
    # Flatten column names
    country_summary.columns = ['Records', 'Total_Value_USD', 'Mean_Value_USD', 'Median_Value_USD', 'Unique_Commodities']
    country_summary = country_summary.sort_values('Total_Value_USD', ascending=False)
    
    st.dataframe(country_summary.reset_index())

    # Export value distribution
    st.subheader("📊 Export Value Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df_sub, x='value_dl', nbins=50,
                          title=f'Export Value Distribution - {sub_region_name}',
                          labels={'value_dl': 'Export Value (USD)', 'count': 'Frequency'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df_sub, y='value_dl',
                     title=f'Export Value Box Plot - {sub_region_name}',
                     labels={'value_dl': 'Export Value (USD)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Country vs Commodity heatmap
    st.subheader("🔥 Top Countries vs Top Commodities Heatmap")
    try:
        # Get top 10 countries and top 10 commodities
        top_countries = df_sub.groupby('country_name')['value_dl'].sum().nlargest(10).index
        top_commodities_list = df_sub.groupby('commodity')['value_dl'].sum().nlargest(10).index
        
        # Filter data for top countries and commodities
        heatmap_data = df_sub[df_sub['country_name'].isin(top_countries) & 
                              df_sub['commodity'].isin(top_commodities_list)]
        
        if not heatmap_data.empty:
            pivot_data = heatmap_data.pivot_table(values='value_dl', 
                                                  index='country_name', 
                                                  columns='commodity', 
                                                  aggfunc='sum').fillna(0)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Export Value (USD)'})
            ax.set_title(f"Export Value Heatmap: Top Countries vs Top Commodities — {sub_region_name}")
            ax.set_xlabel("Commodity")
            ax.set_ylabel("Country")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not create heatmap: {e}")

    return {
        'sub_region_name': sub_region_name,
        'total_records': n_records,
        'total_countries': n_countries,
        'total_value_usd': total_value,
        'mean_value': mean_value,
        'top_countries_by_value': by_value,
        'top_countries_by_count': by_count,
        'top_commodities': top_commodities,
        'category_distribution': category_dist if 'categories' in df_sub.columns else pd.Series(),
        'country_summary': country_summary,
        'raw_data': df_sub
    }

# Load the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/Cleaned_categories.csv')
        return data
    except FileNotFoundError:
        st.error("Please make sure 'Cleaned_categories.csv' is in the data folder.")
        return None

data = load_data()

if data is None:
    st.stop()

# Sidebar for navigation
st.sidebar.title("🚀 Export to European Countries Data Dashboard")
st.sidebar.markdown("---")
options = st.sidebar.selectbox("Select Analysis", [
    "📊 Overview",
    "📅 Top 15 Dates",
    "🌍 Top 15 Countries",
    "🗺️ Top 15 Sub-regions",
    "📦 Top 15 Commodities",
    "🏷️ Top 15 Categories",
    "💰 Value Quantities Analysis",
    "📈 Univariate Analysis",
    "📊 Bivariate Analysis",
    "🔍 Multivariate Analysis",
    "🌎 Geographical Analysis",
    "📉 Time Series Analysis",
    "🥇 Top Commodities Analysis",
    "🎯 Clustering Analysis",
    "🤖 Machine Learning Models",
    "🌏 Country Insights",
    "🌐 Sub-region Insights",
    "⚖️ Country Comparison"
])

# Overview
if options == "📊 Overview":
    st.title("📊 Overview of Export to European Countries Dataset")
    
    # Dataset Introduction
    st.markdown("""
    ### 🌍 About This Dataset
    This dataset contains comprehensive information about **Indian exports to European countries** from 2015 onwards. 
    The data includes detailed export records with commodity information, destination countries, values, and regional classifications.
    
    **Raw Dataset:** `exports-to-european-countries.csv` (Original data)  
    **Processed Dataset:** `Cleaned_categories.csv` (Current analysis data with added categories)
    """)
    
    # Key Metrics Dashboard
    st.subheader("📈 Key Dataset Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        if 'country_name' in data.columns:
            st.metric("Countries", f"{data['country_name'].nunique()}")
        else:
            st.metric("Countries", "N/A")
    with col3:
        if 'commodity' in data.columns:
            st.metric("Commodities", f"{data['commodity'].nunique()}")
        else:
            st.metric("Commodities", "N/A")
    with col4:
        if 'categories' in data.columns:
            st.metric("Categories", f"{data['categories'].nunique()}")
        else:
            st.metric("Categories", "N/A")
    with col5:
        if 'value_dl' in data.columns:
            st.metric("Total Value (USD)", f"${data['value_dl'].sum():,.0f}")
        else:
            st.metric("Total Value", "N/A")
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'sub_region' in data.columns:
            st.metric("Sub-regions", f"{data['sub_region'].nunique()}")
    with col2:
        if 'hs_code' in data.columns:
            st.metric("HS Codes", f"{data['hs_code'].nunique()}")
    with col3:
        if 'date' in data.columns:
            try:
                date_range = pd.to_datetime(data['date'])
                years = date_range.dt.year.nunique()
                st.metric("Years Covered", f"{years}")
            except:
                st.metric("Years Covered", "N/A")
    with col4:
        if 'value_qt' in data.columns:
            total_qty = data['value_qt'].sum()
            st.metric("Total Quantity", f"{total_qty:,.0f}")
    
    st.markdown("---")
    
    # Dataset Structure and Information
    st.subheader("🗂️ Dataset Structure & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Dataset Dimensions:**")
        st.info(f"**Rows:** {data.shape[0]:,} | **Columns:** {data.shape[1]}")
        
        st.write("**📅 Time Period:**")
        if 'date' in data.columns:
            try:
                date_col = pd.to_datetime(data['date'])
                min_date = date_col.min().strftime('%B %Y')
                max_date = date_col.max().strftime('%B %Y')
                st.info(f"**From:** {min_date} **To:** {max_date}")
            except:
                st.info("Date parsing error")
        
        st.write("**🌍 Geographic Coverage:**")
        if 'region' in data.columns and 'sub_region' in data.columns:
            regions = data['region'].nunique()
            sub_regions = data['sub_region'].nunique()
            countries = data['country_name'].nunique()
            st.info(f"**Regions:** {regions} | **Sub-regions:** {sub_regions} | **Countries:** {countries}")
    
    with col2:
        st.write("**📦 Trade Information:**")
        if 'commodity' in data.columns and 'categories' in data.columns:
            commodities = data['commodity'].nunique()
            categories = data['categories'].nunique()
            st.info(f"**Commodities:** {commodities:,} | **Categories:** {categories}")
        
        st.write("**💰 Value Information:**")
        value_cols = ['value_qt', 'value_rs', 'value_dl']
        available_values = [col for col in value_cols if col in data.columns]
        if available_values:
            st.info(f"**Available Values:** {', '.join(available_values)}")
        
        st.write("**📋 Data Completeness:**")
        completeness = ((data.shape[0] * data.shape[1] - data.isnull().sum().sum()) / (data.shape[0] * data.shape[1])) * 100
        st.info(f"**Overall Completeness:** {completeness:.1f}%")
    
    st.markdown("---")
    
    # Column Information
    st.subheader("📋 Column Descriptions")
    
    column_descriptions = {
        'id': 'Unique record identifier',
        'date': 'Export transaction date',
        'country_name': 'Destination country name',
        'alpha_3_code': '3-letter ISO country code',
        'country_code': 'Numeric country code',
        'region': 'Geographic region (e.g., Europe)',
        'region_code': 'Numeric region code',
        'sub_region': 'Geographic sub-region (e.g., Western Europe)',
        'sub_region_code': 'Numeric sub-region code',
        'hs_code': 'Harmonized System commodity code',
        'commodity': 'Commodity/product description',
        'categories': 'Product category classification (added during processing)',
        'unit': 'Unit of measurement (Kgs, Nos, etc.)',
        'value_qt': 'Quantity value',
        'value_rs': 'Value in Indian Rupees',
        'value_dl': 'Value in US Dollars'
    }
    
    col_info_df = pd.DataFrame([
        {
            'Column': col,
            'Description': column_descriptions.get(col, 'Column description not available'),
            'Data Type': str(data[col].dtype),
            'Non-Null Count': f"{data[col].count():,}",
            'Null Count': f"{data[col].isnull().sum():,}",
            'Unique Values': f"{data[col].nunique():,}" if data[col].dtype != 'object' or data[col].nunique() < 1000 else "1000+"
        }
        for col in data.columns
    ])
    
    st.dataframe(col_info_df, use_container_width=True)
    
    st.markdown("---")
    
    # Data Quality Assessment
    st.subheader("🔍 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values Analysis:**")
        missing_data = data.isnull().sum()
        missing_pct = (missing_data / len(data)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        }).sort_values('Missing %', ascending=False)
        
        if missing_df['Missing Count'].sum() > 0:
            st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        else:
            st.success("✅ No missing values found!")
    
    with col2:
        st.write("**Value Distributions:**")
        if 'value_dl' in data.columns:
            st.write(f"**Export Values (USD):**")
            st.write(f"• Min: ${data['value_dl'].min():,.2f}")
            st.write(f"• Max: ${data['value_dl'].max():,.2f}")
            st.write(f"• Mean: ${data['value_dl'].mean():,.2f}")
            st.write(f"• Median: ${data['value_dl'].median():,.2f}")
        
        if 'value_qt' in data.columns:
            st.write(f"**Quantities:**")
            st.write(f"• Min: {data['value_qt'].min():,.2f}")
            st.write(f"• Max: {data['value_qt'].max():,.2f}")
            st.write(f"• Mean: {data['value_qt'].mean():,.2f}")
    
    st.markdown("---")
    
    # Top Categories Preview
    st.subheader("🔝 Quick Data Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'country_name' in data.columns:
            st.write("**Top 10 Countries by Records:**")
            top_countries = data['country_name'].value_counts().head(10)
            st.dataframe(top_countries)
    
    with col2:
        if 'categories' in data.columns:
            st.write("**Top Categories:**")
            top_categories = data['categories'].value_counts().head(10)
            st.dataframe(top_categories)
    
    with col3:
        if 'sub_region' in data.columns:
            st.write("**Top Sub-regions:**")
            top_subregions = data['sub_region'].value_counts().head(10)
            st.dataframe(top_subregions)
    
    st.markdown("---")
    
    # Sample Data Preview
    st.subheader("📋 Sample Data Preview")
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.selectbox("Select sample size:", [5, 10, 20, 50], index=1)
    with col2:
        sort_by = st.selectbox("Sort by:", ['value_dl', 'date', 'country_name', 'commodity'], index=0)
    
    if sort_by in data.columns:
        sample_data = data.nlargest(sample_size, sort_by) if sort_by in ['value_dl', 'value_qt', 'value_rs'] else data.head(sample_size)
    else:
        sample_data = data.head(sample_size)
    
    st.dataframe(sample_data, use_container_width=True)
    
    # Data Processing Information
    st.markdown("---")
    st.subheader("⚙️ Data Processing Information")
    
    st.info("""
    **🔄 Data Processing Steps Applied:**
    
    1. **Raw Data Source:** `exports-to-european-countries.csv` - Original export data
    2. **Category Mapping:** Added product categories based on HS codes and commodity descriptions
    3. **Data Cleaning:** Processed and cleaned for analysis
    4. **Current Dataset:** `Cleaned_categories.csv` - Ready for analysis with 178,202 cleaned records
    
    **📊 Key Transformations:**
    - Added categorical classification for commodities
    - Standardized country and region information
    - Validated and cleaned numerical values
    - Removed duplicate and invalid records
    """)
    
    # Download Information
    st.markdown("---")
    st.subheader("📥 Dataset Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**File Information:**")
        st.write("• Current Dataset: `Cleaned_categories.csv`")
        st.write("• Original Dataset: `exports-to-european-countries.csv`")
        st.write("• Format: CSV (Comma Separated Values)")
        st.write("• Encoding: UTF-8")
    
    with col2:
        st.write("**Usage Guidelines:**")
        st.write("• Use this data for export trend analysis")
        st.write("• Suitable for time series forecasting")
        st.write("• Great for geographic trade analysis")
        st.write("• Perfect for commodity-wise insights")

# Top 15 Dates
elif options == "📅 Top 15 Dates":
    st.title("📅 Top 15 Dates")
    if 'date' in data.columns:
        top_dates = data['date'].value_counts().head(15)
        st.bar_chart(top_dates)
        st.dataframe(top_dates)
    else:
        st.error("Date column not found in dataset")

# Top 15 Countries
elif options == "🌍 Top 15 Countries":
    st.title("🌍 Top 15 Countries")
    if 'country_name' in data.columns:
        top_countries = data['country_name'].value_counts().head(15)
        st.bar_chart(top_countries)
        st.dataframe(top_countries)
    else:
        st.error("Country name column not found in dataset")

# Top 15 Sub-regions
elif options == "🗺️ Top 15 Sub-regions":
    st.title("🗺️ Top 15 Sub-regions")
    if 'sub_region' in data.columns:
        top_subregions = data['sub_region'].value_counts().head(15)
        st.bar_chart(top_subregions)
        st.dataframe(top_subregions)
    else:
        st.error("Sub-region column not found in dataset")

# Top 15 Commodities
elif options == "📦 Top 15 Commodities":
    st.title("📦 Top 15 Commodities")
    if 'commodity' in data.columns:
        top_commodities = data['commodity'].value_counts().head(15)
        st.bar_chart(top_commodities)
        st.dataframe(top_commodities)
    else:
        st.error("Commodity column not found in dataset")

# Top 15 Categories
elif options == "🏷️ Top 15 Categories":
    st.title("🏷️ Top 15 Categories")
    if 'categories' in data.columns:
        top_categories = data['categories'].value_counts().head(15)
        st.bar_chart(top_categories)
        st.dataframe(top_categories)
    else:
        st.error("Categories column not found in dataset")

# Value Quantities Analysis
elif options == "💰 Value Quantities Analysis":
    st.title("💰 Value Quantities Analysis")
    
    # Check which value columns exist
    value_cols = ['value_qt', 'value_rs', 'value_dl']
    available_cols = [col for col in value_cols if col in data.columns]
    
    if available_cols:
        st.write("**Available Value Columns:**", ", ".join(available_cols))
        st.write(data[available_cols].describe())
        
        # Distribution plots
        for col in available_cols:
            st.subheader(f"Distribution of {col}")
            fig, ax = plt.subplots(figsize=(10, 6))
            data[col].dropna().hist(bins=50, ax=ax, alpha=0.7)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    else:
        st.error("No value columns found in dataset")

# Univariate Analysis
elif options == "📈 Univariate Analysis":
    st.title("📈 Univariate Analysis")
    
    # Create dropdown for column selection
    column = st.selectbox("Select Column for Analysis", 
                         ['value_qt', 'value_rs', 'value_dl'])
    
    # Check if the column exists and has numeric data
    if column in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[column].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {column}')
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=data, y=column, ax=ax)
            ax.set_title(f'Box Plot of {column}')
            st.pyplot(fig)
        
        # Show statistics
        st.subheader(f"Statistics for {column}")
        st.write(data[column].describe())
    else:
        st.error(f"Column {column} not found in dataset")

# Bivariate Analysis
elif options == "📊 Bivariate Analysis":
    st.title("📊 Bivariate Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Select X Variable", ['value_qt', 'value_rs', 'value_dl'])
    with col2:
        y_var = st.selectbox("Select Y Variable", ['value_dl', 'value_rs', 'value_qt'])
    
    if x_var != y_var and x_var in data.columns and y_var in data.columns:
        # Sample data for better performance
        sample_data = data[[x_var, y_var]].dropna().sample(min(5000, len(data)))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=sample_data, x=x_var, y=y_var, ax=ax)
        ax.set_title(f'{x_var} vs {y_var}')
        st.pyplot(fig)
        
        # Calculate correlation
        correlation = data[x_var].corr(data[y_var])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        # Heatmap: Top 10 Commodities vs Countries
        if 'value_dl' in data.columns and 'commodity' in data.columns and 'country_name' in data.columns:
            heatmap_data = data.pivot_table(values='value_dl', index='commodity', columns='country_name', aggfunc='sum').fillna(0)
            heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1).sort_values(ascending=False).head(10).index]
            fig_h, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=False, cmap='YlGnBu', linewidths=0.5, ax=ax)
            ax.set_title("Export Value Heatmap (Top 10 Commodities vs Countries)")
            ax.set_xlabel("Country")
            ax.set_ylabel("Commodity")
            plt.tight_layout()
            st.pyplot(fig_h)

        # Heatmap: Top 10 Categories vs Countries
        if 'value_dl' in data.columns and 'categories' in data.columns and 'country_name' in data.columns:
            heatmap_data = data.pivot_table(values='value_dl', index='categories', columns='country_name', aggfunc='sum').fillna(0)
            heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1).sort_values(ascending=False).head(10).index]
            fig_h2, ax2 = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=False, cmap='YlGnBu', linewidths=0.5, ax=ax2)
            ax2.set_title("Export Value Heatmap (Top 10 Categories vs Countries)")
            ax2.set_xlabel("Country")
            ax2.set_ylabel("Category")
            plt.tight_layout()
            st.pyplot(fig_h2)
    else:
        st.warning("Please select different variables for X and Y axes.")

# Multivariate Analysis
elif options == "🔍 Multivariate Analysis":
    st.title("🔍 Multivariate Analysis")
    
    # Select numeric columns for analysis
    numeric_cols = ['value_qt', 'value_rs', 'value_dl']
    available_cols = [col for col in numeric_cols if col in data.columns]
    
    if len(available_cols) >= 2:
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = data[available_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)
        
        # Pairplot with sample data for performance
        st.subheader("Pairwise Relationships")
        sample_size = min(1000, len(data))
        sample_data = data[available_cols].dropna().sample(sample_size)
        
        fig = sns.pairplot(sample_data)
        fig.suptitle("Pairwise Relationships", y=1.02)
        st.pyplot(fig)
    else:
        st.error("Need at least 2 numeric columns for multivariate analysis")

# Geographical Analysis
elif options == "🌎 Geographical Analysis":
    st.title("🌎 Geographical Analysis")
    
    if 'country_name' in data.columns:
        # World Map - Total Export Value by Country
        st.subheader("Total Export Value (USD) by Country - World Map")
        
        if 'alpha_3_code' in data.columns and 'value_dl' in data.columns:
            # Group by country and alpha_3_code, sum the export values
            country_value = data.groupby(['country_name', 'alpha_3_code'])['value_dl'].sum().reset_index()
            
            # Create choropleth map
            fig_map = px.choropleth(country_value,
                                    locations="alpha_3_code",
                                    color="value_dl",
                                    hover_name="country_name",
                                    color_continuous_scale=px.colors.sequential.Plasma,
                                    title="Total Export Value (USD) by Country",
                                    projection="natural earth")
            fig_map.update_layout(height=600)
            st.plotly_chart(fig_map, use_container_width=True)
            
            st.info("💡 **Insight**: This map provides a powerful visual representation of the top export destinations by value. Countries like the USA, UAE, China, and parts of Europe are major importers.")
        else:
            st.warning("World map requires 'alpha_3_code' and 'value_dl' columns")
        
        # Country-wise analysis
        country_analysis = data.groupby('country_name').agg({
            'value_dl': 'sum' if 'value_dl' in data.columns else 'count',
            'value_qt': 'sum' if 'value_qt' in data.columns else 'count'
        }).round(2)
        
        st.subheader("Top 20 Countries by Value")
        top_countries = country_analysis.sort_values(country_analysis.columns[0], ascending=False).head(20)
        st.bar_chart(top_countries[top_countries.columns[0]])
        
        if 'region' in data.columns:
            st.subheader("Regional Analysis")
            regional_data = data.groupby('region')[data.columns[0] if 'value_dl' in data.columns else 'country_name'].sum()
            st.bar_chart(regional_data)
    else:
        st.error("Country information not available for geographical analysis")

# Time Series Analysis
elif options == "📉 Time Series Analysis":
    st.title("📉 Time Series Analysis")
    
    if 'date' in data.columns:
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            
            # Select value column for time series
            value_cols = ['value_rs', 'value_dl', 'value_qt']
            available_cols = [col for col in value_cols if col in data.columns]
            
            if available_cols:
                selected_col = st.selectbox("Select value column for time series", available_cols)
                
                # Monthly aggregation
                monthly_data = data_copy.groupby(data_copy['date'].dt.to_period('M'))[selected_col].sum()
                monthly_data.index = monthly_data.index.to_timestamp()
                
                st.subheader(f"Monthly {selected_col} Time Series")
                st.line_chart(monthly_data)
                
                # Yearly aggregation
                yearly_data = data_copy.groupby(data_copy['date'].dt.year)[selected_col].sum()
                st.subheader(f"Yearly {selected_col} Time Series")
                st.bar_chart(yearly_data)
            else:
                st.error("No value columns available for time series analysis")
        except Exception as e:
            st.error(f"Error in time series analysis: {str(e)}")
    else:
        st.error("Date column not found for time series analysis")

# Top Commodities Analysis
elif options == "🥇 Top Commodities Analysis":
    st.title("🥇 Top Commodities Analysis")
    
    if 'commodity' in data.columns:
        # Select number of top commodities
        top_n = st.slider("Select number of top commodities", 5, 50, 15)
        
        # Select value column
        value_cols = ['value_rs', 'value_dl', 'value_qt']
        available_cols = [col for col in value_cols if col in data.columns]
        
        if available_cols:
            selected_col = st.selectbox("Select value column", available_cols)
            
            # By value
            top_commodities_value = data.groupby('commodity')[selected_col].sum().nlargest(top_n)
            st.subheader(f"Top {top_n} Commodities by {selected_col}")
            st.bar_chart(top_commodities_value)
            st.dataframe(top_commodities_value)
            
            # By frequency
            top_commodities_freq = data['commodity'].value_counts().head(top_n)
            st.subheader(f"Top {top_n} Commodities by Frequency")
            st.bar_chart(top_commodities_freq)
        else:
            st.error("No value columns available")
    else:
        st.error("Commodity column not found")

# Clustering Analysis
elif options == "🎯 Clustering Analysis":
    st.title("🎯 Clustering Analysis")
    
    cluster_type = st.selectbox("Select Clustering Type", 
                               ["Country-wise", "Sub-region-wise", "Value-based"])
    
    if cluster_type == "Country-wise":
        # Country-wise clustering
        if 'country_name' in data.columns:
            country_features = data.groupby('country_name').agg({
                'value_dl': 'sum',
                'value_qt': 'sum',
                'commodity': 'nunique' if 'commodity' in data.columns else 'count'
            }).fillna(0)
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(country_features)
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            country_features['Cluster'] = clusters
            
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(country_features['value_dl'], country_features['value_qt'], 
                               c=clusters, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Total Value (DL)')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Country Clustering')
            plt.colorbar(scatter, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Country column not found")
            
    elif cluster_type == "Sub-region-wise":
        # Sub-region clustering
        if 'sub_region' in data.columns:
            subregion_features = data.groupby('sub_region').agg({
                'value_dl': 'sum',
                'value_qt': 'sum',
                'country_name': 'nunique' if 'country_name' in data.columns else 'count'
            }).fillna(0)
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(subregion_features)
            
            n_clusters = st.slider("Number of Clusters", 2, 8, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            subregion_features['Cluster'] = clusters
            
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(subregion_features['value_dl'], subregion_features['country_name'], 
                               c=clusters, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Total Value (DL)')
            ax.set_ylabel('Number of Countries')
            ax.set_title('Sub-region Clustering')
            plt.colorbar(scatter, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Sub-region column not found")
            
    else:  # Value-based clustering
        # Value-based clustering
        available_cols = ['value_qt', 'value_rs', 'value_dl']
        available_cols = [col for col in available_cols if col in data.columns]
        
        if len(available_cols) >= 2:
            sample_data = data[available_cols].dropna().sample(min(5000, len(data)))
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(sample_data)
            
            n_clusters = st.slider("Number of Clusters", 2, 8, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(sample_data[available_cols[0]], sample_data[available_cols[1]], 
                               c=clusters, cmap='viridis', alpha=0.7)
            ax.set_xlabel(available_cols[0])
            ax.set_ylabel(available_cols[1])
            ax.set_title('Value-based Clustering')
            plt.colorbar(scatter, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Need at least 2 numeric columns for clustering")

# Machine Learning Models
elif options == "🤖 Machine Learning Models":
    st.title("🤖 Machine Learning Models")
    
    model_type = st.selectbox("Select Model Type", 
                             ["Simple Linear Regression", 
                              "Classification Model",
                              "Logistic Regression",
                              "Clustering Model"])
    
    if model_type == "Simple Linear Regression":
        st.subheader("Simple Linear Regression")
        
        # Select features
        numeric_cols = [col for col in ['value_qt', 'value_rs', 'value_dl'] if col in data.columns]
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X Variable", numeric_cols[:-1])
            y_col = st.selectbox("Select Y Variable", [col for col in numeric_cols if col != x_col])
            
            # Prepare data
            model_data = data[[x_col, y_col]].dropna()
            if len(model_data) > 0:
                X = model_data[[x_col]]
                y = model_data[y_col]
                
                # Train model
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Display results
                r2 = r2_score(y, y_pred)
                st.metric("R² Score", f"{r2:.3f}")
                st.metric("Coefficient", f"{model.coef_[0]:.3f}")
                st.metric("Intercept", f"{model.intercept_:.3f}")
                
                # Plot
                sample_size = min(2000, len(model_data))
                sample_data = model_data.sample(sample_size)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(sample_data[x_col], sample_data[y_col], alpha=0.5)
                
                # Add regression line
                x_range = np.linspace(sample_data[x_col].min(), sample_data[x_col].max(), 100)
                y_range = model.predict(x_range.reshape(-1, 1))
                ax.plot(x_range, y_range, 'r-', linewidth=2, label='Regression Line')
                
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Linear Regression: {x_col} vs {y_col}')
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("No valid data available for regression")
        else:
            st.error("Need at least 2 numeric columns for regression")
            
    elif model_type == "Classification Model":
        st.subheader("Classification Model - Predicting Export Categories")
        
        # Check if we have the required columns
        if 'categories' in data.columns and 'value_dl' in data.columns:
            # Prepare data for classification
            classification_data = data[['value_qt', 'value_rs', 'value_dl', 'categories']].dropna()
            
            if len(classification_data) > 100:  # Ensure we have enough data
                # Select features and target
                feature_cols = ['value_qt', 'value_rs', 'value_dl']
                available_features = [col for col in feature_cols if col in classification_data.columns]
                
                if len(available_features) >= 2:
                    X = classification_data[available_features]
                    y = classification_data['categories']
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # Train Random Forest Classifier
                    
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    y_pred = rf_model.predict(X_test)
                    
                    # Display results
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Classification Accuracy", f"{accuracy:.3f}")
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': available_features,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Feature Importance in Category Classification')
                    st.pyplot(fig)
                    
                    # Show top categories predicted
                    st.subheader("Category Distribution in Test Set")
                    category_counts = pd.Series(y_test).value_counts().head(10)
                    st.bar_chart(category_counts)
                    
                else:
                    st.error("Need at least 2 numeric features for classification")
            else:
                st.error("Not enough data for classification (need at least 100 records)")
        else:
            st.error("Classification requires 'categories' and value columns")
            
    elif model_type == "Logistic Regression":
        st.subheader("Logistic Regression - Binary Export Classification")
        numeric_cols = [col for col in ['value_qt', 'value_rs', 'value_dl'] if col in data.columns]
        if len(numeric_cols) >= 2:
            target_col = st.selectbox("Select target column for binary classification", numeric_cols)
            model_data = data[numeric_cols].dropna()
            if len(model_data) > 0:
                median_value = model_data[target_col].median()
                model_data = model_data.copy()
                model_data['target'] = (model_data[target_col] > median_value).astype(int)
                feature_cols = [c for c in numeric_cols if c != target_col]
                X = model_data[feature_cols]
                y = model_data['target']
                if len(X) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Logistic Regression Accuracy", f"{acc:.3f}")
                    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_[0]})
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(coef_df['Feature'], coef_df['Coefficient'])
                    ax.set_xlabel('Coefficient')
                    ax.set_title('Feature Coefficients in Logistic Regression')
                    st.pyplot(fig)
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))
                else:
                    st.error("Not enough data for train-test split")
            else:
                st.error("No valid data for logistic regression")
        else:
            st.error("Need at least 2 numeric columns for logistic regression")
            
    elif model_type == "Clustering Model":
        st.subheader("K-Means Clustering Analysis")
        
        # Select numeric columns for clustering
        numeric_cols = ['value_qt', 'value_rs', 'value_dl']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if len(available_cols) >= 2:
            # Prepare data
            cluster_data = data[available_cols].dropna()
            
            if len(cluster_data) > 0:
                # Sample data for performance if dataset is large
                sample_size = min(5000, len(cluster_data))
                sample_data = cluster_data.sample(sample_size, random_state=42)
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(sample_data)
                
                # User input for number of clusters
                n_clusters = st.slider("Number of Clusters", 2, 10, 4)
                
                # Apply K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Add cluster results to sample data
                sample_data_with_clusters = sample_data.copy()
                sample_data_with_clusters['Cluster'] = clusters
                
                # Visualize clusters
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(sample_data[available_cols[0]], 
                                   sample_data[available_cols[1]], 
                                   c=clusters, cmap='viridis', alpha=0.7)
                ax.set_xlabel(available_cols[0])
                ax.set_ylabel(available_cols[1])
                ax.set_title(f'K-Means Clustering ({n_clusters} clusters)')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)
                
                # Show cluster statistics
                st.subheader("Cluster Statistics")
                cluster_stats = sample_data_with_clusters.groupby('Cluster')[available_cols].mean()
                st.dataframe(cluster_stats.round(2))
                
                # Show cluster sizes
                cluster_sizes = pd.Series(clusters).value_counts().sort_index()
                st.subheader("Cluster Sizes")
                st.bar_chart(cluster_sizes)
                
            else:
                st.error("No valid data available for clustering")
        else:
            st.error("Need at least 2 numeric columns for clustering")
            
    elif model_type == "Simple Linear Regression":
        st.subheader("Simple Linear Regression")
        
        # Select features
        numeric_cols = [col for col in ['value_qt', 'value_rs', 'value_dl'] if col in data.columns]
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X Variable", numeric_cols[:-1])
            y_col = st.selectbox("Select Y Variable", [col for col in numeric_cols if col != x_col])
            
            # Prepare data
            model_data = data[[x_col, y_col]].dropna()
            if len(model_data) > 0:
                X = model_data[[x_col]]
                y = model_data[y_col]
                
                # Train model
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Display results
                r2 = r2_score(y, y_pred)
                st.metric("R² Score", f"{r2:.3f}")
                st.metric("Coefficient", f"{model.coef_[0]:.3f}")
                st.metric("Intercept", f"{model.intercept_:.3f}")
                
                # Plot
                sample_size = min(2000, len(model_data))
                sample_data = model_data.sample(sample_size)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(sample_data[x_col], sample_data[y_col], alpha=0.5)
                
                # Add regression line
                x_range = np.linspace(sample_data[x_col].min(), sample_data[x_col].max(), 100)
                y_range = model.predict(x_range.reshape(-1, 1))
                ax.plot(x_range, y_range, 'r-', linewidth=2, label='Regression Line')
                
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Linear Regression: {x_col} vs {y_col}')
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("No valid data available for regression")
        else:
            st.error("Need at least 2 numeric columns for regression")

# Country Insights Section
elif options == "🌏 Country Insights":
    st.title("🌏 Country Insights Dashboard")
    st.markdown("Select a country from the dropdown to view detailed insights and analytics.")
    
    if 'country_name' in data.columns:
        # Get list of unique countries
        countries = sorted(data['country_name'].dropna().unique())
        
        if countries:
            # Create dropdown for country selection
            selected_country = st.selectbox(
                "🌍 Select a Country for Analysis:",
                options=countries,
                index=0,
                help="Choose a country to view detailed export insights and visualizations"
            )
            
            # Add a button to trigger analysis
            if st.button("🔍 Generate Country Insights", type="primary"):
                with st.spinner(f"Analyzing data for {selected_country}..."):
                    country_insights(selected_country, data)
            
            # Auto-generate insights for the first country on load
            elif selected_country:
                country_insights(selected_country, data)
                
        else:
            st.error("No countries found in the dataset")
    else:
        st.error("Country column not found in the dataset")

# Sub-region Insights Section
elif options == "🌐 Sub-region Insights":
    st.title("🌐 Sub-region Insights Dashboard")
    st.markdown("Select a sub-region from the dropdown to view comprehensive insights, analytics, and visualizations for all countries within that region.")
    
    if 'sub_region' in data.columns:
        # Get list of unique sub-regions
        sub_regions = sorted(data['sub_region'].dropna().unique())
        
        if sub_regions:
            # Create dropdown for sub-region selection
            selected_subregion = st.selectbox(
                "🌍 Select a Sub-region for Analysis:",
                options=sub_regions,
                index=0,
                help="Choose a sub-region to view detailed export insights and visualizations for all countries in that region"
            )
            
            # Display information about the selected sub-region
            if selected_subregion:
                # Show basic info about the sub-region
                subregion_data = data[data['sub_region'] == selected_subregion]
                countries_in_region = sorted(subregion_data['country_name'].unique())
                
                st.info(f"**{selected_subregion}** contains **{len(countries_in_region)}** countries: {', '.join(countries_in_region)}")
                
                # Add a button to trigger analysis
                if st.button("🔍 Generate Sub-region Insights", type="primary"):
                    with st.spinner(f"Analyzing data for {selected_subregion}..."):
                        subregion_insights(selected_subregion, data)
                
                # Auto-generate insights for the first sub-region on load
                else:
                    subregion_insights(selected_subregion, data)
                    
        else:
            st.error("No sub-regions found in the dataset")
    else:
        st.error("Sub-region column not found in the dataset")

# Country Comparison Section
elif options == "⚖️ Country Comparison":
    st.title("⚖️ Country Comparison Dashboard")
    st.markdown("Compare insights and metrics between two countries side by side.")
    
    if 'country_name' in data.columns:
        # Get list of unique countries
        countries = sorted(data['country_name'].dropna().unique())
        
        if len(countries) >= 2:
            # Create two columns for country selection
            col1, col2 = st.columns(2)
            
            with col1:
                country1 = st.selectbox(
                    "🌍 Select First Country:",
                    options=countries,
                    index=0,
                    key="country1"
                )
            
            with col2:
                # Default to second country, or first if only one available
                default_idx = 1 if len(countries) > 1 else 0
                country2 = st.selectbox(
                    "🌎 Select Second Country:",
                    options=countries,
                    index=default_idx,
                    key="country2"
                )
            
            # Comparison button
            if st.button("⚖️ Compare Countries", type="primary"):
                if country1 == country2:
                    st.warning("⚠️ Please select two different countries for comparison.")
                else:
                    with st.spinner(f"Comparing {country1} vs {country2}..."):
                        # Get data for both countries
                        country1_data = country_insights(country1, data)
                        st.markdown("---")
                        country2_data = country_insights(country2, data)
                        
                        if country1_data and country2_data:
                            st.markdown("---")
                            st.subheader(f"📊 Side-by-Side Comparison: {country1} vs {country2}")
                            
                            # Metrics comparison
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Records Comparison",
                                    f"{country1}: {country1_data['total_records']:,}",
                                    f"{country2}: {country2_data['total_records']:,}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Total Value Comparison",
                                    f"{country1}: ${country1_data['total_value_usd']:,.0f}",
                                    f"{country2}: ${country2_data['total_value_usd']:,.0f}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Mean Value Comparison",
                                    f"{country1}: ${country1_data['mean_value']:,.0f}",
                                    f"{country2}: ${country2_data['mean_value']:,.0f}"
                                )
                            
                            with col4:
                                st.metric(
                                    "Median Value Comparison",
                                    f"{country1}: ${country1_data['median_value']:,.0f}",
                                    f"{country2}: ${country2_data['median_value']:,.0f}"
                                )
                            
                            # Comparison visualizations
                            st.subheader("📈 Comparative Visualizations")
                            
                            # Total values comparison
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                comparison_data = pd.DataFrame({
                                    'Country': [country1, country2],
                                    'Total Export Value': [country1_data['total_value_usd'], country2_data['total_value_usd']],
                                    'Total Records': [country1_data['total_records'], country2_data['total_records']]
                                })
                                
                                fig = px.bar(comparison_data, x='Country', y='Total Export Value',
                                           title='Total Export Value Comparison',
                                           color='Country')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.bar(comparison_data, x='Country', y='Total Records',
                                           title='Total Records Comparison',
                                           color='Country')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Top commodities comparison
                            st.subheader("🏆 Top 5 Commodities Comparison")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**{country1} - Top 5 Commodities:**")
                                st.dataframe(country1_data['top_commodities_value'].head().reset_index())
                            
                            with col2:
                                st.write(f"**{country2} - Top 5 Commodities:**")
                                st.dataframe(country2_data['top_commodities_value'].head().reset_index())
                            
        else:
            st.error("Need at least 2 countries in the dataset for comparison")
    else:
        st.error("Country column not found in the dataset")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dataset Info")
st.sidebar.write(f"**Records:** {len(data):,}")
if 'country_name' in data.columns:
    st.sidebar.write(f"**Countries:** {data['country_name'].nunique()}")
if 'categories' in data.columns:
    st.sidebar.write(f"**Categories:** {data['categories'].nunique()}")
if 'date' in data.columns:
    try:
        date_col = pd.to_datetime(data['date'])
        st.sidebar.write(f"**Date Range:** {date_col.min().strftime('%Y-%m-%d')} to {date_col.max().strftime('%Y-%m-%d')}")
    except:
        st.sidebar.write("**Date Range:** Unable to parse dates")

st.sidebar.markdown("---")
st.sidebar.info("🚀 This application provides comprehensive analyses and visualizations of the Cleaned Categories dataset.")