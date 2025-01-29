import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

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

def handle_missing_data(df):
    st.sidebar.write("### Handle Missing Data")
    missing_action = st.sidebar.selectbox(
        "Select Action for Missing Data:",
        ["Leave as is", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Custom Value"]
    )
    
    if missing_action == "Fill with Mean":
        df.fillna(df.mean(), inplace=True)
    elif missing_action == "Fill with Median":
        df.fillna(df.median(), inplace=True)
    elif missing_action == "Fill with Mode":
        df.fillna(df.mode().iloc[0], inplace=True)
    elif missing_action == "Fill with Custom Value":
        custom_value = st.sidebar.text_input("Enter custom value:")
        if custom_value:
            df.fillna(custom_value, inplace=True)
    
    # Display missing data heatmap
    st.write("### Missing Data Heatmap")
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot()

    return df

def detect_and_remove_outliers(df):
    st.sidebar.write("### Outlier Detection & Removal")
    outlier_method = st.sidebar.selectbox("Select Outlier Detection Method:", ["IQR", "Z-score", "None"])
    
    if outlier_method == "IQR":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif outlier_method == "Z-score":
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
        df = df[(z_scores < 3).all(axis=1)]
    
    st.write("### Data After Outlier Removal")
    st.write(df)
    return df

def time_series_analysis(df, date_column):
    st.write("### Time Series Analysis")
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    
    st.write("#### Rolling Average")
    rolling_window = st.slider("Select Rolling Window Size:", 1, 30, 7)
    df_rolling = df.rolling(window=rolling_window).mean()
    fig = px.line(df_rolling, title=f"Rolling Average (Window={rolling_window})")
    st.plotly_chart(fig)

    st.write("#### Seasonal Decomposition")
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df.dropna(), model='additive', period=12)
    fig = decomposition.plot()
    st.pyplot()

def model_comparison(X_train, X_test, y_train, y_test):
    st.write("### Model Comparison")
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        }
    results_df = pd.DataFrame(results).T
    st.write(results_df)

def analyze_and_visualize_file(file):
    try:
        # Determine file type and load data into a Pandas DataFrame
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

        # Handle Missing Data
        df = handle_missing_data(df)

        # Detect and remove outliers
        df = detect_and_remove_outliers(df)

        # Automated Data Insights
        st.write("### Automated Data Insights")
        for col in df.columns:
            st.write(f"Column: {col}")
            st.write(f"Unique Values: {df[col].nunique()}")
            st.write(f"Missing Values: {df[col].isnull().sum()}")
            if df[col].dtype in ['float64', 'int64']:
                st.write(f"Mean: {df[col].mean()}")
                st.write(f"Standard Deviation: {df[col].std()}")
            st.write("-----")

        # Automatically identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        # Let the user choose the type of analysis
        st.sidebar.write("### Customization Options")
        chart_type = st.sidebar.selectbox(
            "Select Chart Type:",
            ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap"]
        )

        if chart_type == "Histogram":
            selected_column = st.sidebar.selectbox(
                "Select Numeric Column for Histogram:",
                numeric_columns
            )
            if selected_column:
                st.write(f"### Distribution of {selected_column}")
                fig = px.histogram(df, x=selected_column, nbins=30, title=f"Distribution of {selected_column}", marginal="box")
                st.plotly_chart(fig)

        elif chart_type == "Boxplot":
            selected_column = st.sidebar.selectbox(
                "Select Numeric Column for Boxplot:",
                numeric_columns
            )
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

        # AI-based Analysis: Clustering
        st.sidebar.write("### AI Analysis")
        if st.sidebar.checkbox("Run Clustering Analysis"):
            clustering_algorithm = st.sidebar.selectbox("Select Clustering Algorithm:", ["K-Means", "DBSCAN", "Agglomerative Clustering"])

            if clustering_algorithm == "K-Means":
                num_clusters = st.sidebar.slider("Select Number of Clusters:", 2, 10, 3)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_columns].dropna())

                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                clusters = kmeans.fit_predict(scaled_data)

            elif clustering_algorithm == "DBSCAN":
                eps = st.sidebar.slider("Select EPS (Epsilon):", 0.1, 10.0, 0.5, 0.1)
                min_samples = st.sidebar.slider("Select Minimum Samples:", 1, 20, 5)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_columns].dropna())

                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(scaled_data)

            elif clustering_algorithm == "Agglomerative Clustering":
                num_clusters = st.sidebar.slider("Select Number of Clusters:", 2, 10, 3)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_columns].dropna())

                agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
                clusters = agg_clustering.fit_predict(scaled_data)

            df['Cluster'] = clusters
            st.write("### Clustering Results")
            st.write(df[['Cluster'] + list(numeric_columns)])

            if len(numeric_columns) >= 2:
                st.write("### Cluster Visualization")
                fig = px.scatter(
                    df, x=numeric_columns[0], y=numeric_columns[1], color="Cluster",
                    title=f"Clusters Visualization ({clustering_algorithm})", template="plotly"
                )
                st.plotly_chart(fig)

        # AI-based Analysis: Predictive Modeling
        if st.sidebar.checkbox("Run Predictive Modeling"):
            st.write("### Predictive Modeling")
            target = st.sidebar.selectbox("Select Target Column:", numeric_columns.union(categorical_columns))

            if target:
                feature_columns = [col for col in numeric_columns if col != target]

                if len(feature_columns) < 1:
                    st.warning("Not enough features for modeling.")
                else:
                    X = df[feature_columns].dropna()
                    y = df[target].dropna()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

                    st.write("#### Feature Importance")
                    importance = model.feature_importances_
                    feature_importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": importance})
                    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
                    fig = px.bar(feature_importance_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
                    st.plotly_chart(fig)

        # Export Option
        if st.sidebar.button("Download Processed Data"):
            df.to_csv('processed_data.csv', index=False)
            st.write("Data saved as 'processed_data.csv'.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit app
st.markdown("<h1 class='main-header'>Excel and CSV Data Analyzer with AI</h1>", unsafe_allow_html=True)
st.write("Upload an Excel or CSV file to analyze and visualize its data with advanced features.")

uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    analyze_and_visualize_file(uploaded_file)
