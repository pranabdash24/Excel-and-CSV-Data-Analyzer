import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 12px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .data-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def analyze_and_visualize_file(file):
    try:
        # Load data into Pandas DataFrame
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            st.error("Unsupported file type. Please upload an Excel or CSV file.")
            return

        st.markdown("<div class='data-section'>", unsafe_allow_html=True)
        st.markdown("### Data Loaded Successfully", unsafe_allow_html=True)
        st.write("#### Data Preview:")
        st.write(df.head())

        st.write("#### Basic Information:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("#### Basic Statistics:")
        st.write(df.describe())
        st.markdown("</div>", unsafe_allow_html=True)

        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        # Sidebar options
        st.sidebar.write("### Customization Options")
        chart_type = st.sidebar.selectbox(
            "Select Chart Type:",
            ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap"]
        )

        if chart_type == "Histogram":
            selected_column = st.sidebar.selectbox("Select Numeric Column for Histogram:", numeric_columns)
            if selected_column:
                st.write(f"### Distribution of {selected_column}")
                fig = px.histogram(df, x=selected_column, nbins=30, title=f"Distribution of {selected_column}", marginal="box")
                st.plotly_chart(fig)

        elif chart_type == "Boxplot":
            selected_column = st.sidebar.selectbox("Select Numeric Column for Boxplot:", numeric_columns)
            if selected_column:
                st.write(f"### Boxplot of {selected_column}")
                fig = px.box(df, y=selected_column, title=f"Boxplot of {selected_column}")
                st.plotly_chart(fig)

        elif chart_type == "Scatter Plot":
            x_axis = st.sidebar.selectbox("Select X-axis:", numeric_columns)
            y_axis = st.sidebar.selectbox("Select Y-axis:", numeric_columns)
            if x_axis and y_axis:
                st.write(f"### Scatter Plot: {x_axis} vs {y_axis}")
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}")
                st.plotly_chart(fig)

        elif chart_type == "Correlation Heatmap" and len(numeric_columns) > 1:
            st.write("### Correlation Heatmap")
            correlation_matrix = df[numeric_columns].corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale="Viridis"
                )
            )
            fig.update_layout(title="Correlation Heatmap")
            st.plotly_chart(fig)

        # Clustering
        st.sidebar.write("### AI Analysis")
        if st.sidebar.checkbox("Run Clustering Analysis"):
            clustering_algorithm = st.sidebar.selectbox("Select Clustering Algorithm:", ["K-Means", "DBSCAN", "Agglomerative Clustering"])
            num_clusters = st.sidebar.slider("Select Number of Clusters:", 2, 10, 3)

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_columns].dropna())

            if clustering_algorithm == "K-Means":
                model = KMeans(n_clusters=num_clusters, random_state=0)
            elif clustering_algorithm == "DBSCAN":
                model = DBSCAN(eps=0.5, min_samples=5)
            else:
                model = AgglomerativeClustering(n_clusters=num_clusters)

            df['Cluster'] = model.fit_predict(scaled_data)
            st.write("### Clustering Results")
            st.write(df[['Cluster'] + list(numeric_columns)])

            if len(numeric_columns) >= 2:
                fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1], color="Cluster", title=f"Clusters Visualization ({clustering_algorithm})")
                st.plotly_chart(fig)

        # Predictive Modeling
        if st.sidebar.checkbox("Run Predictive Modeling"):
            st.write("### Predictive Modeling")
            target = st.sidebar.selectbox("Select Target Column:", numeric_columns.union(categorical_columns))

            if target:
                feature_columns = [col for col in numeric_columns if col != target]

                if len(feature_columns) < 1:
                    st.warning("Not enough features for modeling.")
                else:
                    # Ensure X and y have the same number of samples
                    df_cleaned = df.dropna(subset=[target] + feature_columns)

                    # Define X and y
                    X = df_cleaned[feature_columns]
                    y = df_cleaned[target]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Choose appropriate model
                    if df[target].dtype == 'object' or df[target].dtype.name == 'category':
                        model = RandomForestClassifier(random_state=42)
                    else:
                        model = RandomForestRegressor(random_state=42)

                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    if df[target].dtype == 'object' or df[target].dtype.name == 'category':
                        st.write("#### Classification Report")
                        st.text(classification_report(y_test, predictions))
                    else:
                        st.write("#### Regression Metrics")
                        st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

                    # Feature Importance
                    st.write("#### Feature Importance")
                    importance = model.feature_importances_
                    feature_importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": importance})
                    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
                    fig = px.bar(feature_importance_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
                    st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit app
st.markdown("<h1 class='main-header'>Excel and CSV Data Analyzer with AI</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    analyze_and_visualize_file(uploaded_file)
