import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
# SVC removed as requested
# RandomForestClassifier removed as not in lab plan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, mean_absolute_error


# Custom SVM Implementation from Scratch
class SVMFromScratch:
    """
    Support Vector Machine implemented from scratch using gradient descent.
    Supports binary classification with linear and polynomial kernels.
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, kernel='linear', degree=3, coef0=1):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iterations = n_iterations
        self.kernel = kernel  # 'linear' or 'polynomial'
        self.degree = degree  # Degree for polynomial kernel
        self.coef0 = coef0  # Independent term for polynomial kernel
        self.w = None  # Weights
        self.b = None  # Bias
        self.classes_ = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.X_train = None  # Store training data for kernel methods
    
    def _apply_kernel(self, X):
        """Apply the selected kernel transformation to the input features"""
        if self.kernel == 'linear':
            return X
        elif self.kernel == 'polynomial':
            # Polynomial kernel: (gamma * X + coef0)^degree
            # For simplicity, we'll apply feature transformation
            # K(x, y) = (x·y + coef0)^degree
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            return poly.fit_transform(X)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        
    def fit(self, X, y):
        """Train the SVM model using gradient descent"""
        # Convert to numpy arrays and ensure float dtype
        if hasattr(X, 'values'):
            X = X.values.astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)
            
        if hasattr(y, 'values'):
            y = y.values
        else:
            y = np.array(y)
            
        # Store unique classes
        self.classes_ = np.unique(y)
        
        # For binary classification, convert labels to -1 and 1
        if len(self.classes_) == 2:
            # Map first class to -1, second to 1
            y_binary = np.where(y == self.classes_[0], -1, 1).astype(np.float64)
        else:
            raise ValueError("This SVM implementation only supports binary classification")
        
        # Apply kernel transformation
        X_transformed = self._apply_kernel(X)
        
        n_samples, n_features = X_transformed.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        
        # Gradient descent optimization
        for iteration in range(self.n_iterations):
            for idx in range(n_samples):
                x_i = X_transformed[idx].astype(np.float64)
                condition = y_binary[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # If point is correctly classified beyond margin
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # If point is within margin or misclassified
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - y_binary[idx] * x_i)
                    self.b -= self.learning_rate * y_binary[idx]
        
        # Identify support vectors (points close to decision boundary)
        distances = np.abs(np.dot(X_transformed, self.w) - self.b)
        margin = 1.0 / np.linalg.norm(self.w)
        # Support vectors are points within or on the margin
        self.support_vector_indices_ = np.where(distances <= margin * 1.1)[0]  # 10% tolerance
        self.support_vectors_ = X[self.support_vector_indices_]  # Store original X
        
        # Store the polynomial transformer for prediction
        if self.kernel == 'polynomial':
            self.poly_transformer_ = PolynomialFeatures(degree=self.degree, include_bias=False)
            self.poly_transformer_.fit(X)
        
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        if hasattr(X, 'values'):
            X = X.values
        
        # Apply kernel transformation
        X_transformed = self._apply_kernel(X)
            
        linear_output = np.dot(X_transformed, self.w) - self.b
        y_pred_binary = np.sign(linear_output)
        
        # Convert back to original class labels
        y_pred = np.where(y_pred_binary == -1, self.classes_[0], self.classes_[1])
        
        return y_pred
    
    def decision_function(self, X):
        """Compute the decision function for samples in X"""
        if hasattr(X, 'values'):
            X = X.values
        
        # Apply kernel transformation
        X_transformed = self._apply_kernel(X)
        
        return np.dot(X_transformed, self.w) - self.b
    
    def get_params(self):
        """Return model parameters"""
        return {
            'weights': self.w,
            'bias': self.b,
            'n_support_vectors': len(self.support_vector_indices_) if self.support_vector_indices_ is not None else 0,
            'learning_rate': self.learning_rate,
            'lambda_param': self.lambda_param,
            'n_iterations': self.n_iterations,
            'kernel': self.kernel,
            'degree': self.degree if self.kernel == 'polynomial' else None
        }


# Streamlit page config
st.set_page_config(layout="wide", page_title="ML Model Explorer")

st.title("Machine Learning Mini Project")
st.write("This app implements various Machine Learning models with customizable parameters.")

# Sidebar - data upload
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"]) 

@st.cache_data
def load_sample_data():
    # Generate a larger synthetic dataset for better algorithm testing
    import numpy as np
    np.random.seed(42)  # For reproducible results
    
    n_samples = 300  # Much larger dataset
    
    # Generate synthetic features with realistic patterns
    # Feature 1: Age (normally distributed)
    age = np.random.normal(35, 12, n_samples)
    age = np.clip(age, 18, 80)  # Keep realistic age range
    
    # Feature 2: Income (log-normal distribution)
    income = np.random.lognormal(10.5, 0.5, n_samples)
    income = np.clip(income, 20000, 200000)  # Realistic income range
    
    # Feature 3: Education years (discrete but with noise)
    education_base = np.random.choice([12, 14, 16, 18, 20], n_samples, p=[0.3, 0.2, 0.25, 0.15, 0.1])
    education = education_base + np.random.normal(0, 1, n_samples)
    education = np.clip(education, 10, 25)
    
    # Feature 4: Experience (correlated with age and education)
    experience = np.maximum(0, age - education - 6 + np.random.normal(0, 3, n_samples))
    
    # Feature 5: Score (for regression tasks) - More diverse distribution
    # Create three distinct groups with different score patterns
    group_size = n_samples // 3
    
    # Low performers (score 10-40)
    low_scores = np.random.normal(25, 8, group_size)
    low_scores = np.clip(low_scores, 10, 40)
    
    # Medium performers (score 35-75) 
    medium_scores = np.random.normal(55, 10, group_size)
    medium_scores = np.clip(medium_scores, 35, 75)
    
    # High performers (score 60-95)
    high_scores = np.random.normal(80, 8, group_size + (n_samples % 3))  # Handle remainder
    high_scores = np.clip(high_scores, 60, 95)
    
    # Combine and shuffle all scores
    all_scores = np.concatenate([low_scores, medium_scores, high_scores])
    np.random.shuffle(all_scores)
    score = all_scores
    
    # Create categories for classification (based on score)
    def categorize_performance(score):
        if score < 40:
            return "Low"
        elif score < 70:
            return "Medium" 
        else:
            return "High"
    
    performance = [categorize_performance(s) for s in score]
    
    # Create binary target for additional classification options
    high_performer = ["Yes" if s >= 60 else "No" for s in score]
    
    # Add some categorical features
    departments = np.random.choice(["Sales", "Engineering", "Marketing", "HR", "Finance"], n_samples)
    locations = np.random.choice(["New York", "San Francisco", "Chicago", "Austin", "Remote"], n_samples)
    
    df = pd.DataFrame({
        "Age": age.round(1),
        "Income": income.round(0),
        "Education_Years": education.round(1), 
        "Experience_Years": experience.round(1),
        "Performance_Score": score.round(1),
        "Performance_Category": performance,
        "High_Performer": high_performer,
        "Department": departments,
        "Location": locations
    })
    
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded CSV")
else:
    df = load_sample_data()
    st.sidebar.info("Using sample synthetic dataset")

# Dataset view options
st.subheader("Dataset Overview")
view_option = st.radio("Choose view:", ["Preview (first 10 rows)", "Full Dataset", "Dataset Info"], horizontal=True)

if view_option == "Preview (first 10 rows)":
    st.dataframe(df.head(10))
elif view_option == "Full Dataset":
    st.subheader(f"Complete Dataset ({len(df)} rows × {len(df.columns)} columns)")
    st.dataframe(df)
    
    # Add some dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numerical Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
        
else:  # Dataset Info
    st.subheader("Dataset Information")
    
    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numerical", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical", len(df.select_dtypes(include=['object']).columns))
    with col5:
        missing_cells = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_cells:,}")
    
    st.divider()
    
    # Column Details in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Column Details", "Numerical Summary", "Categorical Summary", "Missing Values"])
    
    with tab1:
        st.write("### Column Information")
        info_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str).values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Null %': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%',
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Memory Usage': [f"{df[col].memory_usage(deep=True) / 1024:.2f} KB" for col in df.columns]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        # Total memory usage
        total_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.caption(f"Total Memory Usage: {total_memory:.2f} MB")
    
    with tab2:
        st.write("### Numerical Columns Statistics")
        numerical_cols = df.select_dtypes(include=[np.number])
        if not numerical_cols.empty:
            stats_df = numerical_cols.describe().T
            stats_df['missing'] = df[numerical_cols.columns].isnull().sum()
            stats_df['missing %'] = (stats_df['missing'] / len(df) * 100).round(2)
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution visualization
            st.write("### Distribution Plots")
            num_cols = numerical_cols.columns.tolist()
            if len(num_cols) > 0:
                selected_col = st.selectbox("Select column to visualize:", num_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(df, x=selected_col, 
                                          title=f"Distribution of {selected_col}",
                                          marginal="box")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(df, y=selected_col,
                                    title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No numerical columns found in the dataset.")
    
    with tab3:
        st.write("### Categorical Columns Summary")
        categorical_cols = df.select_dtypes(include=['object'])
        if not categorical_cols.empty:
            cat_summary = []
            for col in categorical_cols.columns:
                unique_count = df[col].nunique()
                missing_count = df[col].isnull().sum()
                most_frequent = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                most_frequent_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                
                cat_summary.append({
                    'Column Name': col,
                    'Unique Values': unique_count,
                    'Missing': missing_count,
                    'Most Frequent': most_frequent,
                    'Frequency': most_frequent_count
                })
            
            cat_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
            
            # Value counts visualization
            st.write("### Category Distribution")
            selected_cat = st.selectbox("Select categorical column:", categorical_cols.columns.tolist())
            
            value_counts = df[selected_cat].value_counts().head(20)
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Top 20 Values in {selected_cat}",
                           labels={'x': selected_cat, 'y': 'Count'})
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Show unique values for small cardinality
            if df[selected_cat].nunique() <= 20:
                st.write(f"**All unique values:** {', '.join(df[selected_cat].unique().astype(str))}")
        else:
            st.info("No categorical columns found in the dataset.")
    
    with tab4:
        st.write("### Missing Values Analysis")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True, hide_index=True)
            
            # Visualization
            fig_missing = px.bar(missing_data, x='Column', y='Missing %',
                               title="Missing Values by Column",
                               labels={'Missing %': 'Percentage Missing'})
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
            
        # Correlation heatmap for completeness
        st.write("### Missing Value Patterns")
        if len(missing_data) > 0:
            # Show which rows have missing values
            rows_with_missing = df.isnull().any(axis=1).sum()
            st.write(f"**Rows with at least one missing value:** {rows_with_missing} ({rows_with_missing/len(df)*100:.2f}%)")
        else:
            st.write("All rows are complete (no missing values).")

st.divider()  # Add a visual separator

# Sidebar - modeling options
st.sidebar.header("Modeling")
model_type = st.sidebar.selectbox("Select model type", ["Regression", "Classification", "Clustering", "PCA", "Ensemble Learning"]) 

# Common UI for feature/target selection
all_columns = df.columns.tolist()
if model_type in ["Regression", "Classification", "Ensemble Learning"]:
    target = st.sidebar.selectbox("Target column", all_columns)
    features = st.sidebar.multiselect("Feature columns", [c for c in all_columns if c != target], default=[c for c in all_columns if c != target])
else:
    features = st.sidebar.multiselect("Feature columns (for clustering)", all_columns, default=all_columns)

# Model-specific options
if model_type == "Regression":
    model_algo = st.sidebar.selectbox("Algorithm", ["Linear Regression", "Non Linear Regression"])
    
    # Add polynomial degree parameter for non-linear regression
    if model_algo == "Non Linear Regression":
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            poly_degree = st.sidebar.slider ("Non-Linear Degree", 2, 5, 2, key="poly_degree_slider")
        with col2:
            poly_degree = st.sidebar.number_input("", 2, 5, poly_degree, key="poly_degree_input", label_visibility="collapsed")
elif model_type == "Classification":
    model_algo = st.sidebar.selectbox("Algorithm", ["DecisionTree", "SVM"])
    
    # Add SVM hyperparameters
    if model_algo == "SVM":
        st.sidebar.subheader("SVM Parameters")
        svm_kernel = st.sidebar.selectbox("Kernel", ["linear", "polynomial"])
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            svm_learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f", key="svm_lr_slider")
        with col2:
            svm_learning_rate = st.sidebar.number_input("", 0.0001, 0.01, svm_learning_rate, format="%.4f", key="svm_lr_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            svm_lambda = st.sidebar.slider("Regularization (λ)", 0.001, 0.1, 0.01, format="%.3f", key="svm_lambda_slider")
        with col2:
            svm_lambda = st.sidebar.number_input("", 0.001, 0.1, svm_lambda, format="%.3f", key="svm_lambda_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            svm_iterations = st.sidebar.slider("Iterations", 100, 2000, 1000, step=100, key="svm_iter_slider")
        with col2:
            svm_iterations = st.sidebar.number_input("", 100, 2000, svm_iterations, step=100, key="svm_iter_input", label_visibility="collapsed")
        
        # Polynomial kernel specific parameters
        if svm_kernel == "polynomial":
            col1, col2 = st.sidebar.columns([2, 1])
            with col1:
                svm_degree = st.sidebar.slider("Polynomial Degree", 2, 5, 3, key="svm_degree_slider")
            with col2:
                svm_degree = st.sidebar.number_input("", 2, 5, svm_degree, key="svm_degree_input", label_visibility="collapsed")
            
            col1, col2 = st.sidebar.columns([2, 1])
            with col1:
                svm_coef0 = st.sidebar.slider("Coefficient (coef0)", 0.0, 2.0, 1.0, step=0.1, key="svm_coef0_slider")
            with col2:
                svm_coef0 = st.sidebar.number_input("", 0.0, 2.0, svm_coef0, step=0.1, key="svm_coef0_input", label_visibility="collapsed")
elif model_type == "Clustering":
    model_algo = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN"]) 
    
    # PCA preprocessing option
    use_pca = st.sidebar.checkbox("Apply PCA before clustering", value=False, key="clustering_pca")
    
    if use_pca:
        st.sidebar.subheader("PCA Settings")
        # Get numeric columns for PCA
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        pca_features = st.sidebar.multiselect(
            "Select Features for PCA", 
            numeric_columns, 
            default=numeric_columns[:min(5, len(numeric_columns))]
        )
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            pca_components = st.sidebar.slider("PCA Components", 2, min(10, len(pca_features)) if pca_features else 2, 2, key="clustering_pca_components_slider")
        with col2:
            pca_components = st.sidebar.number_input("", 2, min(10, len(pca_features)) if pca_features else 2, pca_components, key="clustering_pca_components_input", label_visibility="collapsed")
    
    # Clustering parameters (always visible when clustering is selected)
    if model_algo == "KMeans":
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, key="kmeans_clusters_slider")
        with col2:
            n_clusters = st.sidebar.number_input("", 2, 10, n_clusters, key="kmeans_clusters_input", label_visibility="collapsed")
    elif model_algo == "DBSCAN":
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5, key="dbscan_eps_slider")
        with col2:
            eps = st.sidebar.number_input("", 0.1, 5.0, eps, key="dbscan_eps_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            min_samples = st.sidebar.slider("Min Samples", 1, 25, 5, key="dbscan_min_samples_slider")
        with col2:
            min_samples = st.sidebar.number_input("", 1, 25, min_samples, key="dbscan_min_samples_input", label_visibility="collapsed")
elif model_type == "PCA":
    st.sidebar.subheader("PCA Parameters")
    
    # Get numeric columns for PCA
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Feature selection for PCA
    feature_cols = st.sidebar.multiselect(
        "Select Features for PCA", 
        numeric_columns, 
        default=numeric_columns[:min(5, len(numeric_columns))]  # Default to first 5 or all if less
    )
    
    # Number of components
    if len(feature_cols) >= 2:
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            n_components = st.sidebar.slider(
                "Number of Components", 
                min_value=2, 
                max_value=len(feature_cols), 
                value=2,
                key="pca_components_slider"
            )
        with col2:
            n_components = st.sidebar.number_input(
                "", 
                min_value=2, 
                max_value=len(feature_cols), 
                value=n_components,
                key="pca_components_input",
                label_visibility="collapsed"
            )
    else:
        n_components = 2
    
    # Column to color plot by
    color_by_column = st.sidebar.selectbox(
        "Select Column to Color Plot By", 
        all_columns,
        index=0
    )
elif model_type == "Ensemble Learning":
    model_algo = st.sidebar.selectbox("Algorithm", ["Random Forest", "AdaBoost", "XGBoost"])
    
    st.sidebar.subheader("Ensemble Parameters")
    
    if model_algo == "Random Forest":
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 500, 100, step=10, key="rf_estimators_slider")
        with col2:
            n_estimators = st.sidebar.number_input("", 10, 500, n_estimators, step=10, key="rf_estimators_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            max_depth = st.sidebar.slider("Max Depth", 1, 30, 10, key="rf_depth_slider")
        with col2:
            max_depth = st.sidebar.number_input("", 1, 30, max_depth, key="rf_depth_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, key="rf_split_slider")
        with col2:
            min_samples_split = st.sidebar.number_input("", 2, 20, min_samples_split, key="rf_split_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1, key="rf_leaf_slider")
        with col2:
            min_samples_leaf = st.sidebar.number_input("", 1, 20, min_samples_leaf, key="rf_leaf_input", label_visibility="collapsed")
        
    elif model_algo == "AdaBoost":
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 50, step=10, key="ada_estimators_slider")
        with col2:
            n_estimators = st.sidebar.number_input("", 10, 500, n_estimators, step=10, key="ada_estimators_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 2.0, 1.0, step=0.01, key="ada_lr_slider")
        with col2:
            learning_rate = st.sidebar.number_input("", 0.01, 2.0, learning_rate, step=0.01, key="ada_lr_input", label_visibility="collapsed")
        
    elif model_algo == "XGBoost":
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100, step=10, key="xgb_estimators_slider")
        with col2:
            n_estimators = st.sidebar.number_input("", 10, 500, n_estimators, step=10, key="xgb_estimators_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            max_depth = st.sidebar.slider("Max Depth", 1, 30, 6, key="xgb_depth_slider")
        with col2:
            max_depth = st.sidebar.number_input("", 1, 30, max_depth, key="xgb_depth_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.3, step=0.01, key="xgb_lr_slider")
        with col2:
            learning_rate = st.sidebar.number_input("", 0.01, 1.0, learning_rate, step=0.01, key="xgb_lr_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            subsample = st.sidebar.slider("Subsample Ratio", 0.5, 1.0, 1.0, step=0.05, key="xgb_subsample_slider")
        with col2:
            subsample = st.sidebar.number_input("", 0.5, 1.0, subsample, step=0.05, key="xgb_subsample_input", label_visibility="collapsed")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            colsample_bytree = st.sidebar.slider("Column Sample Ratio", 0.5, 1.0, 1.0, step=0.05, key="xgb_colsample_slider")
        with col2:
            colsample_bytree = st.sidebar.number_input("", 0.5, 1.0, colsample_bytree, step=0.05, key="xgb_colsample_input", label_visibility="collapsed")

# No additional parameters needed for DecisionTree

run_button = st.sidebar.button("Run")

if run_button:
    X = df[features]
    # handle non-numeric columns
    X = pd.get_dummies(X, drop_first=True)
    
    # Sanitize column names for XGBoost (remove [, ], <, > characters)
    X.columns = X.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False).str.replace('<', '_', regex=False).str.replace('>', '_', regex=False)

    if model_type in ["Regression", "Classification", "Ensemble Learning"]:
        y = df[target]
        # simple split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Regression":
        # Create model based on selection
        if model_algo == "Linear Regression":
            model = LinearRegression()
        elif model_algo == "Non Linear Regression":
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=poly_degree)),
                ('linear', LinearRegression())
            ])
        
        # Train model and make predictions
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        st.subheader(f"{model_algo} Results")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
        with col3:
            st.metric("Mean Absolute Error", f"{mae:.4f}")
        with col4:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Multiple visualizations
        col1, col2 = st.columns(2)
        
        with col1:
                # Actual vs Predicted scatter plot
                fig1 = px.scatter(x=y_test, y=preds, 
                                labels={"x":"Actual Values", "y":"Predicted Values"}, 
                                title="Actual vs Predicted",
                                template=plotly_template)
                # Add perfect prediction line
                min_val = min(min(y_test), min(preds))
                max_val = max(max(y_test), max(preds))
                fig1.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                               mode='lines', name='Perfect Prediction', 
                               line=dict(dash='dash', color='red'))
                st.plotly_chart(fig1)
        
        with col2:
            # Residuals plot
            residuals = y_test - preds
            fig2 = px.scatter(x=preds, y=residuals,
                            labels={"x":"Predicted Values", "y":"Residuals"},
                            title="Residuals Plot",
                            template=plotly_template)
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2)
        
        # Regression "Confusion Matrix" - binned predictions vs actual
        col1, col2 = st.columns(2)
        
        with col1:
                # Create binned confusion matrix for regression
                # Use same threshold for both actual and predicted values
                threshold = np.median(y_test)  # Use actual values median as threshold
                
                # Create binary categories based on the same threshold
                actual_high = y_test >= threshold
                pred_high = preds >= threshold
                
                # Create 2x2 confusion matrix
                regression_cm = np.zeros((2, 2))
                regression_cm[0, 0] = np.sum((~actual_high) & (~pred_high))  # Low-Low
                regression_cm[0, 1] = np.sum((~actual_high) & (pred_high))   # Low-High
                regression_cm[1, 0] = np.sum((actual_high) & (~pred_high))   # High-Low
                regression_cm[1, 1] = np.sum((actual_high) & (pred_high))    # High-High
                
                fig3 = px.imshow(regression_cm,
                               text_auto=True,
                               aspect="auto",
                               title="Regression Confusion Matrix<br>(High/Low Categories)",
                               labels=dict(x="Predicted Category", y="Actual Category"),
                               x=['Low', 'High'],
                               y=['Low', 'High'],
                               color_continuous_scale="Blues")
                st.plotly_chart(fig3)
                
                # Calculate classification metrics for binned categories
                binned_accuracy = (regression_cm[0, 0] + regression_cm[1, 1]) / len(y_test)
                
                # Calculate precision, recall, and F1-score for each class
                # For "Low" class (class 0)
                precision_low = regression_cm[0, 0] / (regression_cm[0, 0] + regression_cm[1, 0]) if (regression_cm[0, 0] + regression_cm[1, 0]) > 0 else 0
                recall_low = regression_cm[0, 0] / (regression_cm[0, 0] + regression_cm[0, 1]) if (regression_cm[0, 0] + regression_cm[0, 1]) > 0 else 0
                f1_low = 2 * (precision_low * recall_low) / (precision_low + recall_low) if (precision_low + recall_low) > 0 else 0
                
                # Calculate overall precision, recall, and F1-score from confusion matrix
                # Using standard formulas for binary classification
                total_true_positives = regression_cm[1, 1]  # High predicted as High
                total_false_positives = regression_cm[0, 1]  # Low predicted as High
                total_false_negatives = regression_cm[1, 0]  # High predicted as Low
                
                precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
                recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Display metrics
                col1_1, col1_2, col1_3, col1_4 = st.columns(4)
                with col1_1:
                    st.metric("Accuracy", f"{binned_accuracy:.4f}")
                with col1_2:
                    st.metric("Precision", f"{precision:.4f}")
                with col1_3:
                    st.metric("Recall", f"{recall:.4f}")
                with col1_4:
                    st.metric("F1-Score", f"{f1_score:.4f}")
                
        
        with col2:
            # Feature importance based on model type
            if model_algo == "Linear Regression":
                # Linear model - show coefficients
                if hasattr(model, 'coef_') and len(X_train.columns) > 1:
                    importance_df = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Coefficient': model.coef_
                    })
                    importance_df = importance_df.reindex(importance_df.Coefficient.abs().sort_values(ascending=False).index)
                    
                    fig4 = px.bar(importance_df, x='Coefficient', y='Feature', 
                                orientation='h', title='Linear Regression Feature Coefficients',
                                template=plotly_template)
                    st.plotly_chart(fig4)
            elif model_algo == "Non Linear Regression":
                # Non-linear - show feature expansion info
                st.write("**Non-Linear Features Created**")
                poly_features = model.named_steps['poly'].get_feature_names_out(X_train.columns)
                st.write(f"Total features: {len(poly_features)}")
                st.write(f"Non-linear degree: {poly_degree}")
                
                # Show top 10 most important non-linear coefficients
                linear_model = model.named_steps['linear']
                if hasattr(linear_model, 'coef_'):
                    coef_df = pd.DataFrame({
                        'Feature': poly_features,
                        'Coefficient': linear_model.coef_
                    })
                    top_coef = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index).head(10)
                    
                    fig4 = px.bar(top_coef, x='Coefficient', y='Feature', 
                                orientation='h', title='Top 10 Non-Linear Feature Coefficients',
                                template=plotly_template)
                    st.plotly_chart(fig4)
        
        # Add 3D visualization for non-linear regression (full-width)
        if model_algo == "Non Linear Regression" and len(X_train.columns) >= 2:
            st.subheader("3D Non-Linear Surface Visualization")
            
            # Use first two features for 3D visualization
            feature1 = X_train.columns[0]
            feature2 = X_train.columns[1]
            
            # Create a grid for surface plotting
            x1_range = np.linspace(X_train[feature1].min(), X_train[feature1].max(), 30)
            x2_range = np.linspace(X_train[feature2].min(), X_train[feature2].max(), 30)
            x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
            
            # Create a DataFrame for prediction with mean values for other features
            grid_df = pd.DataFrame()
            grid_df[feature1] = x1_grid.ravel()
            grid_df[feature2] = x2_grid.ravel()
            
            # Fill other features with their mean values
            for col in X_train.columns:
                if col not in [feature1, feature2]:
                    grid_df[col] = X_train[col].mean()
            
            # Predict on the grid
            grid_predictions = model.predict(grid_df)
            z_grid = grid_predictions.reshape(x1_grid.shape)
            
            # Create 3D surface plot
            fig_3d = px.scatter_3d(
                x=X_test[feature1], 
                y=X_test[feature2], 
                z=y_test,
                title=f'3D Non-Linear Regression Surface<br>Features: {feature1} vs {feature2}',
                labels={
                    'x': feature1,
                    'y': feature2,
                    'z': 'Target Value'
                },
                opacity=0.7,
                template=plotly_template
            )
            
            # Add the non-linear surface
            fig_3d.add_surface(
                x=x1_grid,
                y=x2_grid,
                z=z_grid,
                opacity=0.5,
                colorscale='Viridis',
                name='Non-Linear Surface'
            )
            
            # Update layout for better 3D view
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title=feature1,
                    yaxis_title=feature2,
                    zaxis_title='Target Value',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=700
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Add explanation
            st.info(f"""
            **3D Visualization Explanation:**
            - **Points**: Actual test data points in 3D space
            - **Surface**: Non-linear regression surface (degree {poly_degree})
            - **Features**: Using {feature1} and {feature2} (other features fixed at mean values)
            - **Nonlinearity**: The curved surface shows how non-linear regression captures complex relationships
            """)
        elif model_algo == "Non Linear Regression" and len(X_train.columns) < 2:
            st.info("💡 3D visualization requires at least 2 features in your dataset")


    elif model_type == "Classification":
        # Create model based on selection
        if model_algo == "DecisionTree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_algo == "SVM":
            # Check if binary classification
            n_classes = df[target].nunique()
            
            if n_classes != 2:
                st.error(f"""
                ⚠️ **Binary Classification Required**
                
                The custom SVM implementation only supports **binary classification** (2 classes).
                
                Your selected target **'{target}'** has **{n_classes} classes**: {', '.join(map(str, df[target].unique()))}
                
                **Please select a binary target column**, such as:
                - `High_Performer` (Yes/No)
                - Any column with exactly 2 unique values
                
                Or use **DecisionTree** algorithm which supports multi-class classification.
                """)
                st.stop()
            
            # Create SVM model with kernel parameters
            if svm_kernel == "polynomial":
                model = SVMFromScratch(
                    learning_rate=svm_learning_rate,
                    lambda_param=svm_lambda,
                    n_iterations=svm_iterations,
                    kernel='polynomial',
                    degree=svm_degree,
                    coef0=svm_coef0
                )
            else:
                model = SVMFromScratch(
                    learning_rate=svm_learning_rate,
                    lambda_param=svm_lambda,
                    n_iterations=svm_iterations,
                    kernel='linear'
                )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        st.subheader("Classification Results")
        
        # Metrics
        st.metric("Accuracy", f"{acc:.4f}")
        
        # Visualizations in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Confusion Matrix
            fig1 = px.imshow(cm, 
                           text_auto=True,
                           aspect="auto",
                           title="Confusion Matrix",
                           labels=dict(x="Predicted Class", y="Actual Class"),
                           color_continuous_scale="Blues")
            st.plotly_chart(fig1)
        
        with col2:
            # Classification Report as Bar Chart
            from sklearn.metrics import classification_report
            report = classification_report(y_test, preds, output_dict=True)
            
            # Extract metrics for each class
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            metrics_data = []
            
            for cls in classes:
                metrics_data.extend([
                    {'Class': f'{cls}_precision', 'Value': report[cls]['precision'], 'Metric': 'Precision'},
                    {'Class': f'{cls}_recall', 'Value': report[cls]['recall'], 'Metric': 'Recall'},
                    {'Class': f'{cls}_f1', 'Value': report[cls]['f1-score'], 'Metric': 'F1-Score'}
                ])
            
            metrics_df = pd.DataFrame(metrics_data)
            fig2 = px.bar(metrics_df, x='Class', y='Value', color='Metric',
                         title="Classification Metrics by Class",
                         barmode='group')
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2)
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            fig3 = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                        orientation='h', title='Top 10 Feature Importances')
            st.plotly_chart(fig3)
        
        # Model-specific visualizations
        if model_algo == "DecisionTree":
            st.subheader("Decision Tree Visualization")
            
            # Create matplotlib figure for tree plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            
            # Plot the decision tree
            plot_tree(model, 
                     feature_names=X_train.columns,
                     class_names=model.classes_,
                     filled=True,
                     rounded=True,
                     fontsize=10,
                     max_depth=3,  # Limit depth to keep it readable
                     ax=ax)
            
            ax.set_title("Decision Tree Structure (Limited to Depth 3 for Readability)", fontsize=14)
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close()  # Clean up to avoid memory issues
            
            # Add tree information
            st.info(f"""
            **Tree Information:**
            - Tree Depth: {model.tree_.max_depth}
            - Number of Leaves: {model.tree_.n_leaves}
            - Number of Nodes: {model.tree_.node_count}
            
            *Note: Visualization is limited to depth 3 for readability. The actual tree may be deeper.*
            """)
        
        # SVM-specific visualizations
        if model_algo == "SVM":
            st.subheader("SVM Model Information (Custom Implementation)")
            
            # Get model parameters
            params = model.get_params()
            
            col1, col2 = st.columns(2)
            
            with col1:
                kernel_info = f"**Kernel:** {params['kernel'].capitalize()}"
                if params['kernel'] == 'polynomial':
                    kernel_info += f"\n- Polynomial Degree: {params['degree']}"
                
                st.info(f"""
                **SVM Hyperparameters:**
                {kernel_info}
                - Learning Rate: {params['learning_rate']:.4f}
                - Regularization (λ): {params['lambda_param']:.3f}
                - Training Iterations: {params['n_iterations']}
                - Support Vectors Found: {params['n_support_vectors']}
                """)
            
            with col2:
                st.info(f"""
                **Model Weights:**
                - Weight Vector Norm: {np.linalg.norm(params['weights']):.4f}
                - Bias Term: {params['bias']:.4f}
                - Margin Width: {1.0 / np.linalg.norm(params['weights']):.4f}
                - Support Vector %: {(params['n_support_vectors'] / len(X_train) * 100):.1f}%
                """)
            
            # Decision function visualization (binary classification)
            st.subheader("Decision Function Scores")
            
            decision_scores = model.decision_function(X_test)
            
            # Binary classification visualization
            fig_decision = plt.figure(figsize=(12, 6))
            
            # Plot decision scores
            correct_preds = (preds == y_test.values if hasattr(y_test, 'values') else y_test)
            
            colors_map = ['green' if correct else 'red' for correct in correct_preds]
            plt.scatter(range(len(decision_scores)), decision_scores, 
                      c=colors_map, alpha=0.6, s=50)
            plt.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
            plt.axhline(y=1, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='Margin (+1)')
            plt.axhline(y=-1, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='Margin (-1)')
            plt.xlabel('Test Sample Index', fontsize=12)
            plt.ylabel('Decision Function Score', fontsize=12)
            plt.title('SVM Decision Function Scores\n(Green=Correct, Red=Incorrect)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            st.pyplot(fig_decision)
            plt.close()
            
            st.info("""
            **Decision Function Interpretation:**
            - **Decision Boundary (y=0)**: Separates the two classes
            - **Margins (y=±1)**: Support vectors lie on or within these margins
            - **Green points**: Correctly classified samples
            - **Red points**: Misclassified samples
            - **Distance from boundary**: Indicates classification confidence
            """)
            
            # 2D Decision Boundary Visualization
            if X_train.shape[1] >= 2:
                st.subheader("SVM Decision Boundary (2D Visualization)")
                
                # Use first two features for visualization
                feature1_idx = 0
                feature2_idx = 1
                
                if hasattr(X_train, 'values'):
                    X_train_2d = X_train.values[:, [feature1_idx, feature2_idx]]
                    X_test_2d = X_test.values[:, [feature1_idx, feature2_idx]]
                    feature1_name = X_train.columns[feature1_idx]
                    feature2_name = X_train.columns[feature2_idx]
                else:
                    X_train_2d = X_train[:, [feature1_idx, feature2_idx]]
                    X_test_2d = X_test[:, [feature1_idx, feature2_idx]]
                    feature1_name = f"Feature {feature1_idx}"
                    feature2_name = f"Feature {feature2_idx}"
                
                # Train a 2D version of SVM for visualization
                svm_2d = SVMFromScratch(
                    learning_rate=svm_learning_rate,
                    lambda_param=svm_lambda,
                    n_iterations=svm_iterations
                )
                svm_2d.fit(X_train_2d, y_train)
                
                # Create mesh for decision boundary
                x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
                y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
                
                # Create mesh with fixed number of points to prevent memory issues
                mesh_points = 100  # 100x100 grid
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, mesh_points),
                                   np.linspace(y_min, y_max, mesh_points))
                
                # Predict on mesh
                Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                
                # Convert string labels to numeric for plotting
                if isinstance(Z[0], str):
                    unique_labels = np.unique(Z)
                    label_map = {label: i for i, label in enumerate(unique_labels)}
                    Z_numeric = np.array([label_map[label] for label in Z])
                else:
                    Z_numeric = Z
                
                Z_numeric = Z_numeric.reshape(xx.shape)
                
                # Plot
                fig_boundary = plt.figure(figsize=(12, 8))
                
                # Plot decision boundary and margins
                plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap='RdYlBu')
                plt.contour(xx, yy, Z_numeric, colors='black', linewidths=2, levels=[0.5])
                
                # Get decision function values for margin lines
                decision_values = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
                decision_values = decision_values.reshape(xx.shape)
                
                # Plot margins
                plt.contour(xx, yy, decision_values, colors='blue', 
                          levels=[-1, 0, 1], linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])
                
                # Plot training points
                y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
                
                for i, class_label in enumerate(model.classes_):
                    mask = (y_train_array == class_label)
                    plt.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                              c=['red', 'blue'][i], label=f'Class: {class_label}',
                              s=60, alpha=0.8, edgecolors='black', linewidth=1)
                
                # Highlight support vectors
                if svm_2d.support_vectors_ is not None and len(svm_2d.support_vectors_) > 0:
                    plt.scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1],
                              s=200, facecolors='none', edgecolors='yellow', linewidth=3,
                              label=f'Support Vectors ({len(svm_2d.support_vectors_)})')
                
                plt.xlabel(feature1_name, fontsize=12)
                plt.ylabel(feature2_name, fontsize=12)
                plt.title('SVM Decision Boundary with Margins and Support Vectors', fontsize=14)
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                
                st.pyplot(fig_boundary)
                plt.close()
                
                st.success(f"""
                **Visualization Details:**
                - **Black solid line**: Decision boundary (separates classes)
                - **Blue dashed lines**: Margin boundaries (±1 from decision boundary)
                - **Yellow circles**: Support vectors (critical points defining the boundary)
                - **Colored regions**: Predicted class regions
                - Using features: `{feature1_name}` (x-axis) and `{feature2_name}` (y-axis)
                """)
                
                # Feature weights visualization
                st.subheader("Feature Importance (Weight Magnitudes)")
                
                if hasattr(X_train, 'columns') and params['kernel'] == 'linear':
                    # Only show original feature weights for linear kernel
                    feature_importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Weight': params['weights'],
                        'Absolute Weight': np.abs(params['weights'])
                    }).sort_values('Absolute Weight', ascending=False)
                    
                    fig_weights = px.bar(feature_importance, 
                                        x='Weight', 
                                        y='Feature',
                                        orientation='h',
                                        title='SVM Feature Weights',
                                        labels={'Weight': 'Weight Value'},
                                        color='Weight',
                                        color_continuous_scale='RdBu',
                                        text='Weight')
                    fig_weights.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    st.plotly_chart(fig_weights, use_container_width=True)
                    
                    st.info("""
                    **Feature Weights Interpretation:**
                    - Larger absolute weights = more important features for classification
                    - Positive weights push prediction toward one class
                    - Negative weights push prediction toward the other class
                    """)
                elif params['kernel'] == 'polynomial':
                    st.info(f"""
                    **Polynomial Kernel Feature Space:**
                    
                    With polynomial kernel (degree {params['degree']}), the model operates in a transformed {len(params['weights'])}-dimensional feature space.
                    
                    - Original features: {len(X_train.columns)}
                    - Transformed features: {len(params['weights'])}
                    - The model learns weights for polynomial combinations of features
                    - Feature importance is implicit in the kernel transformation
                    
                    **Note:** Direct feature weights don't map to original features in polynomial kernel space.
                    """)
        
        # Prediction probabilities visualization (if available)
        if pred_proba is not None and len(classes) <= 10:  # Only for reasonable number of classes
            prob_df = pd.DataFrame(pred_proba, columns=[f'Class_{cls}' for cls in classes])
            prob_df['Actual'] = y_test.values
            prob_df['Predicted'] = preds
            prob_df['Sample'] = range(len(prob_df))
            
            # Show prediction probabilities for first 20 samples
            fig4 = px.bar(prob_df.head(20).melt(id_vars=['Sample', 'Actual', 'Predicted'], 
                                               value_vars=[col for col in prob_df.columns if col.startswith('Class_')]),
                         x='Sample', y='value', color='variable',
                         title='Prediction Probabilities (First 20 samples)',
                         labels={'value': 'Probability', 'variable': 'Class'})
            st.plotly_chart(fig4)

    elif model_type == "Ensemble Learning":  # Ensemble Learning
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
        try:
            import xgboost as xgb
        except ImportError:
            st.error("XGBoost is not installed. Please install it using: pip install xgboost")
            st.stop()
        
        # Determine if this is a classification or regression problem
        is_classification = df[target].dtype == 'object' or df[target].nunique() < 20
        
        st.subheader(f"{model_algo} - {'Classification' if is_classification else 'Regression'}")
        
        # Create model based on selection
        if model_algo == "Random Forest":
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
        elif model_algo == "AdaBoost":
            if is_classification:
                model = AdaBoostClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
            else:
                model = AdaBoostRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
                
        elif model_algo == "XGBoost":
            if is_classification:
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42
                )
        
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        if is_classification:
            # Classification metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            acc = accuracy_score(y_test, preds)
            
            # Metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
            with col2:
                st.metric("Test Samples", len(y_test))
            with col3:
                st.metric("Number of Classes", len(np.unique(y_train)))
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig_cm = px.imshow(cm, 
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=model.classes_,
                             y=model.classes_,
                             text_auto=True,
                             color_continuous_scale='Blues',
                             template=plotly_template)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, preds, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score']))
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(importance_df.head(20), 
                               x='Importance', 
                               y='Feature',
                               orientation='h',
                               title=f'Top 20 Feature Importances ({model_algo})',
                               template=plotly_template)
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            
            # Predictions vs Actual
            st.subheader("Predictions vs Actual")
            pred_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': preds
            })
            pred_df['Correct'] = pred_df['Actual'] == pred_df['Predicted']
            pred_df['Sample'] = range(len(pred_df))
            
            fig_pred = px.scatter(pred_df.head(100), 
                                x='Sample', 
                                y='Actual',
                                color='Correct',
                                title='Predictions vs Actual (First 100 samples)',
                                hover_data=['Predicted'],
                                template=plotly_template)
            st.plotly_chart(fig_pred, use_container_width=True)
            
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.4f}")
            with col3:
                st.metric("Mean Absolute Error", f"{mae:.4f}")
            with col4:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(importance_df.head(20), 
                               x='Importance', 
                               y='Feature',
                               orientation='h',
                               title=f'Top 20 Feature Importances ({model_algo})',
                               template=plotly_template)
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            
            # Actual vs Predicted scatter plot
            st.subheader("Actual vs Predicted Values")
            fig1 = px.scatter(x=y_test, y=preds, 
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title=f'{model_algo} - Actual vs Predicted',
                            template=plotly_template)
            # Add diagonal line for perfect predictions
            min_val = min(y_test.min(), preds.min())
            max_val = max(y_test.max(), preds.max())
            fig1.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                           mode='lines', name='Perfect Prediction',
                           line=dict(dash='dash', color='red'))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Residuals plot
            st.subheader("Residuals Plot")
            residuals = y_test - preds
            fig2 = px.scatter(x=preds, y=residuals,
                            labels={'x': 'Predicted Values', 'y': 'Residuals'},
                            title='Residuals vs Predicted Values',
                            template=plotly_template)
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Distribution of residuals
            st.subheader("Residuals Distribution")
            fig3 = px.histogram(residuals, nbins=30,
                              labels={'value': 'Residuals', 'count': 'Frequency'},
                              title='Distribution of Residuals',
                              template=plotly_template)
            st.plotly_chart(fig3, use_container_width=True)

    elif model_type == "Clustering":  # Clustering
        # Prepare data for clustering
        if use_pca:
            # Use PCA preprocessing
            if len(pca_features) < 2:
                st.warning("Please select at least 2 numeric features for PCA preprocessing.")
                st.stop()
            
            # Get PCA features and standardize
            X_pca_input = df[pca_features]
            if X_pca_input.isnull().sum().sum() > 0:
                st.warning("Selected PCA features contain missing values. Please clean the data first.")
                st.stop()
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pca_input)
            
            # Apply PCA
            pca = PCA(n_components=pca_components)
            X_cluster = pca.fit_transform(X_scaled)
            
            # Create feature names for PCA components
            feature_names = [f'PC{i+1}' for i in range(pca_components)]
            
            st.info(f"Applied PCA: Reduced from {len(pca_features)} features to {pca_components} components. "
                   f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            # Use original features
            X_cluster = pd.get_dummies(df[features], drop_first=True)
            feature_names = X_cluster.columns.tolist()
        
        # Apply clustering algorithm
        if model_algo == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)

        labels = model.fit_predict(X_cluster)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points for DBSCAN
        
        st.subheader("Clustering Results")
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clusters Found", n_clusters_found)
        with col2:
            noise_points = list(labels).count(-1) if -1 in labels else 0
            st.metric("Noise Points", noise_points)
        
        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        st.write("**Cluster Distribution:**")
        st.write(cluster_counts)
        
        # Multiple visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # 2D visualization
            if use_pca and pca_components >= 2:
                # Already have PCA coordinates, just use first 2 components
                plot_df_2d = pd.DataFrame(X_cluster[:, :2], columns=["PC1", "PC2"])
                plot_df_2d["Cluster"] = labels.astype(str)
                
                fig1 = px.scatter(plot_df_2d, x="PC1", y="PC2", color="Cluster", 
                                title=f"Clusters in PCA Space\nPC1 vs PC2 ({pca.explained_variance_ratio_[:2].sum():.3f} variance)")
                st.plotly_chart(fig1)
            else:
                # Apply PCA for visualization only
                if isinstance(X_cluster, pd.DataFrame):
                    pca_viz = PCA(n_components=2)
                    coords_2d = pca_viz.fit_transform(X_cluster)
                else:
                    # X_cluster is already numpy array from PCA
                    coords_2d = X_cluster[:, :2] if X_cluster.shape[1] >= 2 else X_cluster
                    pca_viz = None
                
                plot_df_2d = pd.DataFrame(coords_2d, columns=["PC1", "PC2"]) 
                plot_df_2d["Cluster"] = labels.astype(str)
                
                if pca_viz:
                    title = f"Clusters in 2D (PCA)\nExplained Variance: {pca_viz.explained_variance_ratio_.sum():.3f}"
                else:
                    title = "Clusters in 2D (PCA Components)"
                
                fig1 = px.scatter(plot_df_2d, x="PC1", y="PC2", color="Cluster", title=title)
                st.plotly_chart(fig1)
        
        with col2:
            # Cluster size distribution
            fig2 = px.bar(x=cluster_counts.index.astype(str), y=cluster_counts.values,
                         labels={'x': 'Cluster', 'y': 'Number of Points'},
                         title="Cluster Size Distribution")
            st.plotly_chart(fig2)
        
        # 3D visualization if we have enough components
        if X_cluster.shape[1] >= 3:
            if use_pca and pca_components >= 3:
                # Already have PCA coordinates, use first 3 components
                plot_df_3d = pd.DataFrame(X_cluster[:, :3], columns=["PC1", "PC2", "PC3"])
                plot_df_3d["Cluster"] = labels.astype(str)
                
                fig3 = px.scatter_3d(plot_df_3d, x="PC1", y="PC2", z="PC3", color="Cluster",
                                   title=f"Clusters in PCA Space (3D)\nPC1-PC3 ({pca.explained_variance_ratio_[:3].sum():.3f} variance)")
                st.plotly_chart(fig3)
            else:
                # Apply PCA for 3D visualization
                if isinstance(X_cluster, pd.DataFrame):
                    pca_3d = PCA(n_components=3)
                    coords_3d = pca_3d.fit_transform(X_cluster)
                else:
                    coords_3d = X_cluster[:, :3] if X_cluster.shape[1] >= 3 else X_cluster
                    pca_3d = None
                
                plot_df_3d = pd.DataFrame(coords_3d, columns=["PC1", "PC2", "PC3"])
                plot_df_3d["Cluster"] = labels.astype(str)
                
                if pca_3d:
                    title = f"Clusters in 3D (PCA)\nExplained Variance: {pca_3d.explained_variance_ratio_.sum():.3f}"
                else:
                    title = "Clusters in 3D (PCA Components)"
                
                fig3 = px.scatter_3d(plot_df_3d, x="PC1", y="PC2", z="PC3", color="Cluster", title=title)
                st.plotly_chart(fig3)
        
        # Clustering evaluation metrics
        if len(set(labels)) > 1:  # Need at least 2 clusters for meaningful metrics
            from sklearn.metrics import silhouette_score, silhouette_samples
            
            silhouette_avg = silhouette_score(X_cluster, labels)
            
            st.subheader("Clustering Evaluation Metrics")
            
            # Metrics in columns
            if model_algo == "KMeans":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                with col2:
                    st.metric("Inertia (WCSS)", f"{model.inertia_:.2f}")
                with col3:
                    st.metric("Number of Iterations", model.n_iter_)
            else:  # DBSCAN
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                with col2:
                    noise_ratio = list(labels).count(-1) / len(labels) if -1 in labels else 0
                    st.metric("Noise Ratio", f"{noise_ratio:.2%}")
            
            # Silhouette plot
            sample_silhouette_values = silhouette_samples(X_cluster, labels)
            silhouette_df = pd.DataFrame({
                'Silhouette_Score': sample_silhouette_values,
                'Cluster': labels
            })
            
            fig4 = px.histogram(silhouette_df, x='Silhouette_Score', color='Cluster',
                              title="Silhouette Score Distribution by Cluster",
                              nbins=30)
            fig4.add_vline(x=silhouette_avg, line_dash="dash", 
                          annotation_text=f"Average: {silhouette_avg:.3f}")
            st.plotly_chart(fig4)
        
        # Feature importance for clustering
        if use_pca:
            # Show PCA component importance
            st.subheader("PCA Components Used for Clustering")
            component_importance = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(pca_components)],
                'Explained_Variance': pca.explained_variance_ratio_
            })
            
            fig5 = px.bar(component_importance, x='Explained_Variance', y='Component',
                         orientation='h', title="PCA Component Contribution to Clustering")
            st.plotly_chart(fig5)
            
            # Show original feature loadings if available
            if hasattr(pca, 'components_'):
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                loadings_df = pd.DataFrame(loadings, 
                                          columns=[f'PC{i+1}' for i in range(pca_components)],
                                          index=pca_features)
                st.write("**Original Feature Contributions to PCA Components:**")
                st.dataframe(loadings_df.round(3))
        else:
            # Show feature variance contribution (original approach)
            if isinstance(X_cluster, pd.DataFrame):
                feature_variance = X_cluster.var().sort_values(ascending=False)
                fig5 = px.bar(x=feature_variance.values, y=feature_variance.index,
                             orientation='h', title="Feature Variance (Clustering Contribution)")
                st.plotly_chart(fig5)

    elif model_type == "PCA":  # PCA
        st.header("PCA Results")
        
        # Check if at least 2 features are selected
        if len(feature_cols) < 2:
            st.warning("Please select at least 2 numeric features for PCA.")
            st.stop()
        
        # Get the data for selected features
        X = df[feature_cols]
        
        # Check for missing values
        if X.isnull().sum().sum() > 0:
            st.warning("Selected features contain missing values. Please clean the data first.")
            st.stop()
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Show Explained Variance
        explained_variance = pca.explained_variance_ratio_
        st.subheader("Explained Variance")
        st.write(f"Total Variance Explained by {n_components} components: {explained_variance.sum():.2%}")
        
        # Explained variance bar chart
        variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Explained Variance': explained_variance
        })
        
        fig_variance = px.bar(variance_df, x='Component', y='Explained Variance',
                             title="Explained Variance by Principal Component",
                             labels={'Explained Variance': 'Explained Variance Ratio'})
        st.plotly_chart(fig_variance, use_container_width=True)
        
        # Show cumulative variance
        cumulative_variance = np.cumsum(explained_variance)
        fig_cumulative = px.line(x=range(1, len(cumulative_variance)+1), y=cumulative_variance,
                                title="Cumulative Explained Variance",
                                labels={'x': 'Number of Components', 'y': 'Cumulative Variance'})
        fig_cumulative.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                annotation_text="80% Variance")
        fig_cumulative.add_hline(y=0.95, line_dash="dash", line_color="orange",
                                annotation_text="95% Variance")
        st.plotly_chart(fig_cumulative, use_container_width=True)
        
        # Show 2D Scatter Plot
        st.subheader("PCA Visualization")
        
        # Create PCA DataFrame
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        
        # Add the color column from original dataframe
        pca_df[color_by_column] = df[color_by_column].values
        
        # Create 2D scatter plot
        fig_2d = px.scatter(pca_df, x="PC1", y="PC2", color=color_by_column,
                           title="2D PCA Scatter Plot",
                           labels={'PC1': f'PC1 ({explained_variance[0]:.1%} variance)',
                                  'PC2': f'PC2 ({explained_variance[1]:.1%} variance)'})
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # If we have 3 or more components, show 3D plot
        if n_components >= 3:
            fig_3d = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color=color_by_column,
                                  title="3D PCA Scatter Plot",
                                  labels={'PC1': f'PC1 ({explained_variance[0]:.1%})',
                                         'PC2': f'PC2 ({explained_variance[1]:.1%})',
                                         'PC3': f'PC3 ({explained_variance[2]:.1%})'})
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Show feature loadings
        st.subheader("Feature Loadings")
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        loadings_df = pd.DataFrame(loadings, 
                                  columns=[f'PC{i+1}' for i in range(n_components)],
                                  index=feature_cols)
        
        # Display loadings table
        st.write("**Feature contributions to each Principal Component:**")
        st.dataframe(loadings_df.round(3))
        
        # Loadings plot for first two components
        if n_components >= 2:
            fig_loadings = px.scatter(x=loadings_df['PC1'], y=loadings_df['PC2'],
                                     text=loadings_df.index,
                                     title="Feature Loadings Plot (PC1 vs PC2)",
                                     labels={'x': 'PC1 Loadings', 'y': 'PC2 Loadings'})
            fig_loadings.update_traces(textposition="top center")
            fig_loadings.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_loadings.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_loadings, use_container_width=True)
        
        # Show PCA summary
        st.subheader("PCA Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Components", n_components)
        with col2:
            st.metric("Original Features", len(feature_cols))
        with col3:
            st.metric("Variance Retained", f"{explained_variance.sum():.1%}")

else:
    st.info("Configure options in the sidebar and click Run to train models.")