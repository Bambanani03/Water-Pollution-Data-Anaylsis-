import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Water Pollution Analysis",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Water Pollution Analysis Dashboard")
st.markdown(
    """
    ### Data Science, Machine Learning and Water Quality Monitoring

    This dashboard analyses water-quality data using:
    - 📊 Exploratory Data Analysis
    - 🌲 Random Forest classification
    - 🔵 K-Means clustering
    - 📈 Prophet time-series forecasting
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("china_water_pollution_data.csv")

    return data


# Sidebar
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload water pollution CSV",
    type=["csv"]
)

try:
    df = load_data(uploaded_file)
except FileNotFoundError:
    st.error(
        "CSV file not found. Please upload "
        "`china_water_pollution_data.csv` using the sidebar."
    )
    st.stop()


# Convert date
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🌲 Random Forest",
        "🔵 K-Means Clustering",
        "📈 Prophet Forecast"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Water Pollution Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Columns",
            f"{len(df.columns)}"
        )

    with col3:
        st.metric(
            "Average pH",
            f"{df['pH'].mean():.2f}"
        )

    with col4:
        st.metric(
            "Average WQI",
            f"{df['Water_Quality_Index'].mean():.2f}"
        )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Number of records:**", len(df))
        st.write("**Number of columns:**", len(df.columns))

    with col2:
        st.write(
            "**Missing values:**",
            int(df.isnull().sum().sum())
        )
        st.write(
            "**Duplicate rows:**",
            int(df.duplicated().sum())
        )

    st.info(
        "The dataset contains water-quality measurements such as "
        "pH, dissolved oxygen, turbidity, nutrients, COD, BOD, "
        "heavy metals, coliform count and Water Quality Index."
    )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.header("📊 Exploratory Data Analysis")

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    st.sidebar.subheader("Data Filters")

    filtered_df = df.copy()

    if "Province" in df.columns:

        provinces = st.sidebar.multiselect(
            "Select Province",
            sorted(df["Province"].dropna().unique()),
            default=[]
        )

        if provinces:
            filtered_df = filtered_df[
                filtered_df["Province"].isin(provinces)
            ]

    if "City" in df.columns:

        cities = st.sidebar.multiselect(
            "Select City",
            sorted(df["City"].dropna().unique()),
            default=[]
        )

        if cities:
            filtered_df = filtered_df[
                filtered_df["City"].isin(cities)
            ]

    st.write(
        f"Showing **{len(filtered_df):,}** records."
    )

    # --------------------------------------------------------
    # pH distribution
    # --------------------------------------------------------

    st.subheader("📈 pH Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        filtered_df["pH"].dropna(),
        bins=30
    )

    ax.axvline(
        6.5,
        linestyle="--",
        label="pH = 6.5"
    )

    ax.set_xlabel("pH")
    ax.set_ylabel("Number of Records")
    ax.set_title("Distribution of Water pH")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------------
    # WQI
    # --------------------------------------------------------

    st.subheader("💧 Water Quality Index")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        filtered_df["Water_Quality_Index"].dropna(),
        bins=30
    )

    ax.set_xlabel("Water Quality Index")
    ax.set_ylabel("Number of Records")
    ax.set_title("Water Quality Index Distribution")

    st.pyplot(fig)

    # --------------------------------------------------------
    # Pollution level
    # --------------------------------------------------------

    if "Pollution_Level" in filtered_df.columns:

        st.subheader("🚨 Pollution Levels")

        pollution_counts = (
            filtered_df["Pollution_Level"]
            .value_counts()
        )

        st.bar_chart(pollution_counts)

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    st.subheader("🔗 Correlation Matrix")

    numeric_df = filtered_df.select_dtypes(
        include=np.number
    )

    if not numeric_df.empty:

        correlation = numeric_df.corr()

        st.dataframe(
            correlation.round(2),
            use_container_width=True
        )


# ============================================================
# RANDOM FOREST
# ============================================================

elif page == "🌲 Random Forest":

    st.header("🌲 Random Forest – Low pH Prediction")

    st.write(
        """
        The Random Forest model predicts whether a water sample
        belongs to the **Unsafe Low pH** class.

        In the project, low pH is defined as:

        **pH < 6.5 → Unsafe Low pH**
        """
    )

    # Create target
    model_df = df.copy()

    model_df["Unsafe_Low_pH"] = (
        model_df["pH"] < 6.5
    ).astype(int)

    # Features from project
    features = [
        "Water_Temperature_C",
        "Dissolved_Oxygen_mg_L",
        "Conductivity_uS_cm",
        "Turbidity_NTU",
        "Nitrate_mg_L",
        "Nitrite_mg_L",
        "Ammonia_N_mg_L",
        "Total_Phosphorus_mg_L",
        "Total_Nitrogen_mg_L",
        "COD_mg_L",
        "BOD_mg_L",
        "Heavy_Metals_Pb_ug_L",
        "Heavy_Metals_Cd_ug_L",
        "Heavy_Metals_Hg_ug_L",
        "Coliform_Count_CFU_100mL",
        "Water_Quality_Index"
    ]

    # Keep only features that exist
    features = [
        feature for feature in features
        if feature in model_df.columns
    ]

    X = model_df[features].copy()
    y = model_df["Unsafe_Low_pH"]

    # Fill missing numerical values
    X = X.fillna(X.median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Training Records",
            f"{len(X_train):,}"
        )

    with col3:
        st.metric(
            "Testing Records",
            f"{len(X_test):,}"
        )

    st.divider()

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader("🎯 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest Confusion Matrix")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Safe", "Unsafe"]
    )

    ax.set_yticklabels(
        ["Safe", "Unsafe"]
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    st.subheader("⭐ Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    # --------------------------------------------------------
    # Prediction Interface
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔮 Test a New Water Sample")

    input_values = {}

    cols = st.columns(2)

    for index, feature in enumerate(features):

        with cols[index % 2]:

            default_value = float(
                X[feature].median()
            )

            input_values[feature] = st.number_input(
                feature,
                value=default_value
            )

    if st.button(
        "🔍 Predict Water Quality",
        type="primary"
    ):

        input_df = pd.DataFrame(
            [input_values]
        )

        prediction = rf_model.predict(
            input_df
        )[0]

        probability = rf_model.predict_proba(
            input_df
        )[0]

        if prediction == 1:

            st.error(
                "⚠️ Prediction: UNSAFE LOW pH"
            )

            st.write(
                f"Model probability: "
                f"{probability[1] * 100:.2f}%"
            )

        else:

            st.success(
                "✅ Prediction: SAFE pH"
            )

            st.write(
                f"Model probability: "
                f"{probability[0] * 100:.2f}%"
            )


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

elif page == "🔵 K-Means Clustering":

    st.header("🔵 K-Means Water Quality Clustering")

    st.write(
        """
        K-Means groups water samples with similar characteristics.

        The project uses **5 clusters** based on water-quality
        variables.
        """
    )

    cluster_features = [
        "pH",
        "Dissolved_Oxygen_mg_L",
        "Conductivity_uS_cm",
        "Turbidity_NTU",
        "COD_mg_L",
        "BOD_mg_L",
        "Water_Quality_Index"
    ]

    cluster_features = [
        feature for feature in cluster_features
        if feature in df.columns
    ]

    X_cluster = df[cluster_features].copy()

    # Fill missing values
    X_cluster = X_cluster.fillna(
        X_cluster.median()
    )

    # Standardization
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_cluster
    )

    # K-Means
    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        X_scaled
    )

    cluster_df = df.copy()

    cluster_df["Cluster"] = clusters

    # --------------------------------------------------------
    # Assign status based on average pH
    # --------------------------------------------------------

    cluster_pH = (
        cluster_df
        .groupby("Cluster")["pH"]
        .mean()
        .sort_values()
    )

    status_labels = {
        cluster_pH.index[0]: "Very Unsafe",
        cluster_pH.index[1]: "Unsafe",
        cluster_pH.index[2]: "Moderate",
        cluster_pH.index[3]: "Safe",
        cluster_pH.index[4]: "Very Safe"
    }

    cluster_df["KMeans_Status"] = (
        cluster_df["Cluster"]
        .map(status_labels)
    )

    # --------------------------------------------------------
    # Cluster summary
    # --------------------------------------------------------

    st.subheader("📊 Cluster Summary")

    summary = (
        cluster_df
        .groupby(
            ["Cluster", "KMeans_Status"]
        )
        .agg(
            Samples=("pH", "count"),
            Average_pH=("pH", "mean"),
            Average_WQI=(
                "Water_Quality_Index",
                "mean"
            )
        )
        .reset_index()
    )

    st.dataframe(
        summary.round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Cluster counts
    # --------------------------------------------------------

    st.subheader("📈 Number of Samples per Cluster")

    cluster_counts = (
        cluster_df["KMeans_Status"]
        .value_counts()
    )

    st.bar_chart(cluster_counts)

    # --------------------------------------------------------
    # pH by cluster
    # --------------------------------------------------------

    st.subheader("💧 Average pH by Cluster")

    avg_ph = (
        cluster_df
        .groupby("KMeans_Status")["pH"]
        .mean()
        .sort_values()
    )

    st.bar_chart(avg_ph)

    # --------------------------------------------------------
    # Show data
    # --------------------------------------------------------

    st.subheader("📋 Clustered Data")

    st.dataframe(
        cluster_df[
            [
                "pH",
                "Water_Quality_Index",
                "Cluster",
                "KMeans_Status"
            ]
        ].head(50),
        use_container_width=True
    )


# ============================================================
# PROPHET FORECAST
# ============================================================

elif page == "📈 Prophet Forecast":

    st.header("📈 pH Time-Series Forecast")

    if not PROPHET_AVAILABLE:

        st.error(
            "Prophet is not installed."
        )

        st.code(
            "pip install prophet"
        )

        st.stop()

    st.write(
        """
        Prophet is used to analyse historical pH values over time
        and produce a future forecast.
        """
    )

    # --------------------------------------------------------
    # Prepare time-series data
    # --------------------------------------------------------

    ts_data = df[
        ["Date", "pH"]
    ].copy()

    ts_data = ts_data.dropna()

    ts_data = (
        ts_data
        .groupby("Date")["pH"]
        .mean()
        .reset_index()
    )

    ts_data = ts_data.rename(
        columns={
            "Date": "ds",
            "pH": "y"
        }
    )

    ts_data = ts_data.sort_values(
        "ds"
    )

    # --------------------------------------------------------
    # Historical pH
    # --------------------------------------------------------

    st.subheader("📊 Historical pH")

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        ts_data["ds"],
        ts_data["y"]
    )

    ax.axhline(
        6.5,
        linestyle="--",
        label="pH = 6.5"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("pH")
    ax.set_title("Historical Water pH")

    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Forecast days
    # --------------------------------------------------------

    forecast_days = st.slider(
        "Number of days to forecast",
        min_value=7,
        max_value=90,
        value=30
    )

    # --------------------------------------------------------
    # Train Prophet
    # --------------------------------------------------------

    with st.spinner(
        "Training Prophet model..."
    ):

        prophet_model = Prophet()

        prophet_model.fit(
            ts_data
        )

        future = prophet_model.make_future_dataframe(
            periods=forecast_days,
            freq="D"
        )

        forecast = prophet_model.predict(
            future
        )

    # --------------------------------------------------------
    # Forecast chart
    # --------------------------------------------------------

    st.subheader(
        f"🔮 {forecast_days}-Day pH Forecast"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    historical = forecast[
        forecast["ds"] <= ts_data["ds"].max()
    ]

    future_forecast = forecast[
        forecast["ds"] > ts_data["ds"].max()
    ]

    ax.plot(
        historical["ds"],
        historical["yhat"],
        label="Historical/Fitted"
    )

    ax.plot(
        future_forecast["ds"],
        future_forecast["yhat"],
        linestyle="--",
        label="Forecast"
    )

    ax.fill_between(
        future_forecast["ds"],
        future_forecast["yhat_lower"],
        future_forecast["yhat_upper"],
        alpha=0.2,
        label="Uncertainty"
    )

    ax.axhline(
        6.5,
        linestyle=":",
        label="pH = 6.5"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("pH")
    ax.set_title(
        "Water pH Forecast"
    )

    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Forecast table
    # --------------------------------------------------------

    st.subheader("📋 Forecast Results")

    forecast_table = future_forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ].copy()

    forecast_table.columns = [
        "Date",
        "Predicted pH",
        "Lower Bound",
        "Upper Bound"
    ]

    st.dataframe(
        forecast_table.round(3),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Forecast interpretation
    # --------------------------------------------------------

    average_forecast = (
        future_forecast["yhat"].mean()
    )

    st.metric(
        "Average Forecasted pH",
        f"{average_forecast:.2f}"
    )

    if average_forecast < 6.5:

        st.warning(
            "⚠️ The average forecasted pH is below 6.5. "
            "This indicates a potential low-pH period "
            "that should be investigated."
        )

    else:

        st.success(
            "✅ The average forecasted pH is above 6.5."
        )

    st.info(
        "Forecasts are estimates and should support, "
        "not replace, real-world water testing and "
        "expert environmental assessment."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(
    "💧 Water Pollution Analysis Project\n\n"
    "Machine Learning + Data Science + Forecasting"
)

st.caption(
    "Water Pollution Data Analysis Dashboard"
)