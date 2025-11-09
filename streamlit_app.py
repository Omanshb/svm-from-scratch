import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from svm_from_scratch import (
    SVMClassifier,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


def load_sample_datasets():
    """Load built-in classification datasets for demonstration."""
    datasets = {}
    
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        datasets['Iris (Classification)'] = {
            'data': pd.DataFrame(iris.data, columns=iris.feature_names),
            'target': iris.target,
            'target_names': iris.target_names,
            'description': 'Iris dataset - 3 classes, 4 features'
        }
        
        datasets['Iris Binary (Classification)'] = {
            'data': pd.DataFrame(iris.data[iris.target != 2], columns=iris.feature_names),
            'target': iris.target[iris.target != 2],
            'target_names': iris.target_names[:2],
            'description': 'Iris dataset (setosa vs versicolor) - 2 classes, 4 features'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        datasets['Breast Cancer (Classification)'] = {
            'data': pd.DataFrame(cancer.data, columns=cancer.feature_names),
            'target': cancer.target,
            'target_names': cancer.target_names,
            'description': 'Breast cancer diagnostic dataset - binary classification'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import load_wine
        wine = load_wine()
        datasets['Wine (Classification)'] = {
            'data': pd.DataFrame(wine.data, columns=wine.feature_names),
            'target': wine.target,
            'target_names': wine.target_names,
            'description': 'Wine dataset - 3 classes, 13 features'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                                   n_redundant=0, n_classes=2, n_clusters_per_class=1,
                                   class_sep=1.5, random_state=42)
        datasets['Synthetic 2D (Classification)'] = {
            'data': pd.DataFrame(X, columns=['Feature 1', 'Feature 2']),
            'target': y,
            'target_names': np.array(['Class 0', 'Class 1']),
            'description': 'Synthetic 2D dataset - perfect for visualizing decision boundaries'
        }
    except ImportError:
        pass
    
    return datasets


def create_confusion_matrix_plot(cm, class_names):
    """Create confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[str(c) for c in class_names],
        y=[str(c) for c in class_names],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500
    )
    
    return fig


def create_decision_boundary_plot(X, y, model, feature_names):
    """Create enhanced decision boundary plot for 2D data."""
    if X.shape[1] != 2:
        return None
    
    # Create a fine mesh for smooth visualization
    h = 0.01
    margin = 1.0
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions and decision function values
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    try:
        Z_decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        if len(Z_decision.shape) > 1:
            Z_decision = Z_decision[:, 0]
        Z_decision = Z_decision.reshape(xx.shape)
    except:
        Z_decision = None
    
    fig = go.Figure()
    
    # Add decision boundary as a filled contour with gradient
    if Z_decision is not None:
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z_decision,
            colorscale='RdYlBu',
            showscale=True,
            opacity=0.6,
            contours=dict(
                start=-3,
                end=3,
                size=0.5,
            ),
            colorbar=dict(
                title="Decision<br>Function",
                titleside="right",
                tickmode="linear",
                tick0=-3,
                dtick=1
            ),
            hovertemplate='<b>Decision Value: %{z:.2f}</b><br>' +
                          f'{feature_names[0]}: %{{x:.2f}}<br>' +
                          f'{feature_names[1]}: %{{y:.2f}}<extra></extra>'
        ))
    else:
        # Fallback to simple class prediction
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale='RdYlBu',
            showscale=True,
            opacity=0.6,
            colorbar=dict(title="Predicted<br>Class")
        ))
    
    # Add data points with different colors for each class
    unique_classes = np.unique(y)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, class_val in enumerate(unique_classes):
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(
                size=10,
                color=colors[idx % len(colors)],
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            hovertemplate=f'<b>Class {class_val}</b><br>' +
                          f'{feature_names[0]}: %{{x:.2f}}<br>' +
                          f'{feature_names[1]}: %{{y:.2f}}<extra></extra>'
        ))
    
    # Highlight support vectors
    support_vectors = model.get_support_vectors()
    if support_vectors is not None and not isinstance(support_vectors, list):
        fig.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            name='Support Vectors',
            marker=dict(
                size=16,
                color='rgba(255, 215, 0, 0.3)',
                symbol='circle',
                line=dict(width=3, color='gold')
            ),
            hovertemplate='<b>Support Vector</b><br>' +
                          f'{feature_names[0]}: %{{x:.2f}}<br>' +
                          f'{feature_names[1]}: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'SVM Decision Boundary Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        width=800,
        height=600,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis=dict(gridcolor='white', gridwidth=2),
        yaxis=dict(gridcolor='white', gridwidth=2),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    return fig


def create_probability_distribution_plot(y_true, y_proba, n_classes):
    """Create probability distribution plot for the predicted class."""
    if n_classes == 2:
        df_plot = pd.DataFrame({
            'Probability': y_proba[:, 1],
            'True Class': [f'Class {y}' for y in y_true]
        })
        
        fig = px.histogram(
            df_plot, x='Probability', color='True Class',
            nbins=30, barmode='overlay',
            title='Predicted Probability Distribution (Class 1)',
            labels={'Probability': 'Predicted Probability for Class 1'}
        )
        
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Decision Threshold")
    else:
        max_proba = np.max(y_proba, axis=1)
        df_plot = pd.DataFrame({
            'Max Probability': max_proba,
            'True Class': [f'Class {y}' for y in y_true]
        })
        
        fig = px.histogram(
            df_plot, x='Max Probability', color='True Class',
            nbins=30, barmode='overlay',
            title='Maximum Predicted Probability Distribution',
            labels={'Max Probability': 'Maximum Predicted Probability'}
        )
    
    fig.update_layout(width=700, height=500)
    return fig


def create_roc_curve_plot(y_true, y_proba, n_classes):
    """Create ROC curve plot for binary classification."""
    if n_classes != 2:
        return None
    
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    auc = np.trapz(tpr_list, fpr_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr_list, y=tpr_list,
        mode='lines',
        name=f'ROC Curve (AUC = {abs(auc):.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    
    return fig


def create_class_distribution_plot(y_train, y_test, target_names):
    """Create bar plot showing class distribution in train and test sets."""
    train_counts = np.bincount(y_train.astype(int))
    test_counts = np.bincount(y_test.astype(int))
    
    df_plot = pd.DataFrame({
        'Class': list(target_names) * 2,
        'Count': list(train_counts) + list(test_counts),
        'Set': ['Train'] * len(train_counts) + ['Test'] * len(test_counts)
    })
    
    fig = px.bar(
        df_plot, x='Class', y='Count', color='Set',
        barmode='group',
        title='Class Distribution in Train and Test Sets'
    )
    
    fig.update_layout(width=600, height=400)
    return fig


def create_margin_plot(X, y, model, feature_names):
    """Create plot showing decision boundary and margins for 2D binary classification."""
    if X.shape[1] != 2 or model.n_classes_ != 2:
        return None
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='RdBu',
        showscale=True,
        contours=dict(
            start=-2,
            end=2,
            size=0.5,
        ),
        colorbar=dict(title="Decision Function")
    ))
    
    unique_classes = np.unique(y)
    for class_val in unique_classes:
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(size=8, line=dict(width=1, color='white'))
        ))
    
    support_vectors = model.get_support_vectors()
    if support_vectors is not None and not isinstance(support_vectors, list):
        fig.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            name='Support Vectors',
            marker=dict(
                size=12,
                color='yellow',
                symbol='circle-open',
                line=dict(width=3, color='black')
            )
        ))
    
    fig.update_layout(
        title='SVM Decision Function and Margins',
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        width=700,
        height=500
    )
    
    return fig


def create_3d_decision_surface(X, y, model, feature_names):
    """Create interactive 3D visualization of SVM decision surface."""
    # Use PCA if we have more than 3 features
    if X.shape[1] > 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
        feature_labels = [f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                         f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                         f'PC3 ({pca.explained_variance_ratio_[2]:.1%})']
        st.info(f"Using PCA to reduce {X.shape[1]} features to 3D for visualization. " +
                f"Explained variance: {pca.explained_variance_ratio_[:3].sum():.1%}")
    elif X.shape[1] == 3:
        X_3d = X
        feature_labels = feature_names[:3]
    elif X.shape[1] == 2:
        # For 2D data, create a 3D visualization with decision function as Z-axis
        X_3d = np.column_stack([X, model.decision_function(X)])
        feature_labels = [feature_names[0], feature_names[1], 'Decision Function']
    else:
        return None
    
    # Create mesh grid for decision surface
    resolution = 20
    x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
    y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # For decision surface, we need to predict on the grid
    if X.shape[1] == 2:
        # Original 2D data
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.decision_function(grid_points).reshape(xx.shape)
    else:
        # For higher dimensions, we'll use the mean values for other dimensions
        grid_points = []
        for i in range(len(xx.ravel())):
            point = np.mean(X_3d, axis=0).copy()
            point[0] = xx.ravel()[i]
            point[1] = yy.ravel()[i]
            grid_points.append(point)
        grid_points = np.array(grid_points)
        
        # Transform back if we used PCA
        if X.shape[1] > 3:
            grid_points_original = pca.inverse_transform(grid_points)
        else:
            grid_points_original = grid_points
        
        try:
            Z_vals = model.decision_function(grid_points_original)
            if len(Z_vals.shape) > 1:
                Z_vals = Z_vals[:, 0]
            Z = Z_vals.reshape(xx.shape)
        except:
            Z = model.predict(grid_points_original).reshape(xx.shape)
    
    # Create figure
    fig = go.Figure()
    
    # Add decision surface
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=Z,
        colorscale='RdYlBu',
        opacity=0.6,
        name='Decision Surface',
        showscale=True,
        colorbar=dict(title="Decision<br>Value", x=1.1),
        hovertemplate=f'{feature_labels[0]}: %{{x:.2f}}<br>' +
                      f'{feature_labels[1]}: %{{y:.2f}}<br>' +
                      'Decision: %{z:.2f}<extra></extra>'
    ))
    
    # Add data points
    unique_classes = np.unique(y)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, class_val in enumerate(unique_classes):
        mask = y == class_val
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(
                size=5,
                color=colors[idx % len(colors)],
                line=dict(width=1, color='white'),
                opacity=0.9
            ),
            hovertemplate=f'<b>Class {class_val}</b><br>' +
                          f'{feature_labels[0]}: %{{x:.2f}}<br>' +
                          f'{feature_labels[1]}: %{{y:.2f}}<br>' +
                          f'{feature_labels[2]}: %{{z:.2f}}<extra></extra>'
        ))
    
    # Highlight support vectors if available
    support_vectors = model.get_support_vectors()
    if support_vectors is not None and not isinstance(support_vectors, list):
        # Transform support vectors to 3D space if needed
        if X.shape[1] > 3:
            sv_3d = pca.transform(support_vectors)
        elif X.shape[1] == 3:
            sv_3d = support_vectors
        elif X.shape[1] == 2:
            sv_3d = np.column_stack([support_vectors, model.decision_function(support_vectors)])
        
        fig.add_trace(go.Scatter3d(
            x=sv_3d[:, 0],
            y=sv_3d[:, 1],
            z=sv_3d[:, 2],
            mode='markers',
            name='Support Vectors',
            marker=dict(
                size=8,
                color='gold',
                symbol='diamond',
                line=dict(width=2, color='black'),
                opacity=1.0
            ),
            hovertemplate='<b>Support Vector</b><br>' +
                          f'{feature_labels[0]}: %{{x:.2f}}<br>' +
                          f'{feature_labels[1]}: %{{y:.2f}}<br>' +
                          f'{feature_labels[2]}: %{{z:.2f}}<extra></extra>'
        ))
    
    # Update layout for better 3D viewing
    fig.update_layout(
        title={
            'text': 'Interactive 3D SVM Decision Surface',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2C3E50'}
        },
        scene=dict(
            xaxis=dict(title=feature_labels[0], backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            yaxis=dict(title=feature_labels[1], backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            zaxis=dict(title=feature_labels[2] if X.shape[1] >= 3 else 'Decision Value',
                      backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=900,
        height=700,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="SVM from Scratch",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Support Vector Machine from Scratch")
    st.markdown("Implementation using Sequential Minimal Optimization (SMO) with Lagrange multipliers")
    
    st.sidebar.header("Data Selection")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Built-in Datasets", "Upload CSV"]
    )
    
    df = None
    target = None
    target_names = None
    dataset_name = ""
    
    if data_source == "Built-in Datasets":
        datasets = load_sample_datasets()
        
        if not datasets:
            st.error("No built-in datasets available. Please install scikit-learn.")
            return
        
        dataset_choice = st.sidebar.selectbox(
            "Select dataset:",
            list(datasets.keys())
        )
        
        if dataset_choice:
            dataset = datasets[dataset_choice]
            df = dataset['data']
            target = dataset['target']
            target_names = dataset['target_names']
            dataset_name = dataset_choice
            
            st.sidebar.info(f"**{dataset_choice}**\n\n{dataset['description']}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name
                
                if len(df.columns) < 2:
                    st.sidebar.error("Dataset must have at least 2 columns (features + target)")
                    df = None
                    target = None
                else:
                    target_col = st.sidebar.selectbox(
                        "Select target column:",
                        list(df.columns)
                    )
                    
                    if target_col:
                        target = df[target_col].values
                        df = df.drop(columns=[target_col])
                        unique_classes = np.unique(target)
                        target_names = unique_classes
                        
                        # Check if target has at least 2 classes
                        if len(unique_classes) < 2:
                            st.sidebar.error("Target must have at least 2 different classes for classification")
                            df = None
                            target = None
            except Exception as e:
                st.sidebar.error(f"Error reading CSV file: {str(e)}")
                df = None
                target = None
    
    if df is not None and target is not None:
        st.header(f"Dataset: {dataset_name}")
        
        # Data quality summary
        with st.expander("📋 Data Quality Checks", expanded=False):
            checks = []
            checks.append("✓ Dataset loaded successfully")
            
            if df.isnull().any().any():
                checks.append("⚠️ Contains missing values (will be handled)")
            else:
                checks.append("✓ No missing values")
            
            if np.isinf(df.select_dtypes(include=[np.number]).values).any():
                checks.append("❌ Contains infinite values")
            else:
                checks.append("✓ No infinite values")
            
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                checks.append(f"⚠️ Non-numeric columns: {', '.join(non_numeric)}")
            else:
                checks.append("✓ All features are numeric")
            
            for check in checks:
                st.text(check)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            n_classes = len(np.unique(target))
            st.metric("Classes", n_classes)
        
        with st.expander("Dataset Preview"):
            preview_df = df.copy()
            preview_df['Target'] = target
            st.dataframe(preview_df.head(10))
        
        st.subheader("Model Configuration")
        
        feature_selection = st.multiselect(
            "Select features (leave empty for all):",
            list(df.columns),
            default=[]
        )
        
        if len(feature_selection) == 0:
            feature_selection = list(df.columns)
        
        # Check for non-numeric features
        non_numeric_cols = df[feature_selection].select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.error(f"❌ Non-numeric features detected: {', '.join(non_numeric_cols)}")
            st.error("SVM requires numeric features. Please encode categorical variables or remove them.")
            st.stop()
        
        # Check for missing values
        if df[feature_selection].isnull().any().any():
            st.warning("⚠️ Missing values detected. Rows with missing values will be removed.")
            mask = ~df[feature_selection].isnull().any(axis=1)
            df = df[mask]
            target = target[mask]
            st.info(f"Dataset reduced to {len(df)} samples after removing missing values.")
        
        # Check for infinite values
        if np.isinf(df[feature_selection].values).any():
            st.error("❌ Infinite values detected in features. Please clean your data.")
            st.stop()
        
        # Check for constant features (zero variance)
        constant_cols = df[feature_selection].columns[df[feature_selection].nunique() == 1].tolist()
        if constant_cols:
            st.warning(f"⚠️ Constant features detected and will be removed: {', '.join(constant_cols)}")
            feature_selection = [col for col in feature_selection if col not in constant_cols]
            if len(feature_selection) == 0:
                st.error("❌ No variable features remaining after removing constant features.")
                st.stop()
        
        # Check if we have enough samples
        if len(df) < 10:
            st.error("❌ Not enough samples (minimum 10 required after cleaning)")
            st.stop()
        
        X = df[feature_selection].values
        y = target
        
        # Validate target variable
        try:
            y = np.array(y)
            if not np.issubdtype(y.dtype, np.number):
                # Try to convert to numeric if possible
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.info(f"Target variable encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        except Exception as e:
            st.error(f"❌ Error processing target variable: {str(e)}")
            st.stop()
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        
        # Check class distribution for stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            # If stratification fails (e.g., too few samples per class), try without stratification
            st.warning(f"⚠️ Cannot stratify split: {str(e)}. Using random split instead.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        scale_features = st.checkbox(
            "Scale features (recommended for SVM)",
            value=True,
            help="SVM is sensitive to feature scales. Scaling is highly recommended."
        )
        
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        st.markdown("#### SVM Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            kernel = st.selectbox(
                "Kernel",
                ["linear", "rbf", "poly"],
                help="Kernel function: linear, RBF (Radial Basis Function), or polynomial"
            )
        
        with col2:
            C = st.slider(
                "C (Regularization)",
                0.01, 10.0, 1.0, 0.1,
                help="Regularization strength. Smaller C = stronger regularization (softer margin)"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_iterations = st.slider(
                "Max Iterations",
                100, 2000, 1000, 100,
                help="Maximum iterations for SMO algorithm"
            )
        
        with col2:
            tol = st.select_slider(
                "Tolerance",
                options=[1e-4, 1e-3, 1e-2],
                value=1e-3,
                format_func=lambda x: f"{x:.0e}",
                help="Convergence tolerance for optimization"
            )
        
        # Kernel-specific parameters
        if kernel == "rbf":
            gamma = st.slider(
                "Gamma (RBF)",
                0.01, 1.0, 0.1, 0.01,
                help="Kernel coefficient. Higher gamma = more complex decision boundary"
            )
            degree = 3
        elif kernel == "poly":
            degree = st.slider(
                "Degree (Polynomial)",
                2, 5, 3, 1,
                help="Degree of polynomial kernel"
            )
            gamma = 0.1
        else:
            gamma = 0.1
            degree = 3
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training SVM model using SMO..."):
                try:
                    model = SVMClassifier(
                        kernel=kernel,
                        C=C,
                        n_iterations=n_iterations,
                        gamma=gamma,
                        degree=degree,
                        tol=tol,
                        random_state=42
                    )
                    
                    model.fit(X_train, y_train)
                    
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    y_train_proba = model.predict_proba(X_train)
                    y_test_proba = model.predict_proba(X_test)
                    
                    st.session_state['model'] = model
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['y_train_pred'] = y_train_pred
                    st.session_state['y_test_pred'] = y_test_pred
                    st.session_state['y_train_proba'] = y_train_proba
                    st.session_state['y_test_proba'] = y_test_proba
                    st.session_state['feature_names'] = feature_selection
                    st.session_state['target_names'] = target_names
                    st.session_state['n_classes'] = n_classes
                    
                    st.success("Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    st.error("Try adjusting the parameters or checking your data.")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            y_train_pred = st.session_state['y_train_pred']
            y_test_pred = st.session_state['y_test_pred']
            y_train_proba = st.session_state['y_train_proba']
            y_test_proba = st.session_state['y_test_proba']
            feature_names = st.session_state['feature_names']
            target_names = st.session_state['target_names']
            n_classes = st.session_state['n_classes']
            
            st.header("Model Results")
            
            average_type = 'binary' if n_classes == 2 else 'macro'
            
            train_acc = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average=average_type)
            train_recall = recall_score(y_train, y_train_pred, average=average_type)
            train_f1 = f1_score(y_train, y_train_pred, average=average_type)
            
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average=average_type)
            test_recall = recall_score(y_test, y_test_pred, average=average_type)
            test_f1 = f1_score(y_test, y_test_pred, average=average_type)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Metrics")
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Accuracy", f"{train_acc:.4f}")
                    st.metric("Precision", f"{train_precision:.4f}")
                with met_col2:
                    st.metric("Recall", f"{train_recall:.4f}")
                    st.metric("F1 Score", f"{train_f1:.4f}")
            
            with col2:
                st.subheader("Test Metrics")
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Accuracy", f"{test_acc:.4f}")
                    st.metric("Precision", f"{test_precision:.4f}")
                with met_col2:
                    st.metric("Recall", f"{test_recall:.4f}")
                    st.metric("F1 Score", f"{test_f1:.4f}")
            
            st.markdown("---")
            
            with st.expander("Model Details"):
                n_support = model.get_n_support()
                
                st.write("**Support Vectors:**")
                if n_support is not None:
                    if isinstance(n_support, list):
                        for i, n in enumerate(n_support):
                            st.write(f"- Class {target_names[i]}: {n} support vectors")
                        st.write(f"- Total: {sum(n_support)} support vectors")
                    else:
                        st.write(f"- {n_support} support vectors identified")
                    st.caption("Support vectors are training points where α > 0 (Lagrange multipliers)")
                
                st.write("")
                st.write("**Configuration:**")
                params = model.get_params()
                for key, value in params.items():
                    if isinstance(value, float):
                        st.write(f"- {key}: {value:.4g}")
                    else:
                        st.write(f"- {key}: {value}")
            
            st.subheader("Visualizations")
            
            tab_names = ["3D Decision Surface", "Confusion Matrix", "Probability Distribution", "Class Distribution"]
            
            if n_classes == 2:
                tab_names.append("ROC Curve")
            
            if len(feature_names) == 2:
                tab_names.append("Decision Boundary")
                if n_classes == 2:
                    tab_names.append("Margins")
            
            viz_tabs = st.tabs(tab_names)
            
            tab_idx = 0
            
            # 3D Decision Surface Tab
            with viz_tabs[tab_idx]:
                try:
                    fig_3d = create_3d_decision_surface(X_test, y_test, model, feature_names)
                    if fig_3d:
                        st.plotly_chart(fig_3d, width='stretch')
                        
                        st.info("""
                        **3D Decision Surface:**
                        - Colored surface represents the SVM decision function in 3D space
                        - Data points are colored by their true class
                        - Gold diamonds indicate support vectors
                        - Click and drag to rotate, scroll to zoom
                        """)
                    else:
                        st.warning("3D visualization not available for this configuration.")
                except Exception as e:
                    st.error(f"Error creating 3D visualization: {str(e)}")
            tab_idx += 1
            
            with viz_tabs[tab_idx]:
                cm = confusion_matrix(y_test, y_test_pred)
                fig_cm = create_confusion_matrix_plot(cm, target_names)
                st.plotly_chart(fig_cm, width='stretch')
                
                st.info("""
                **Confusion Matrix:**
                - Shows the distribution of actual vs predicted classifications
                - Diagonal elements show correct predictions
                - Off-diagonal elements show misclassifications
                """)
            tab_idx += 1
            
            with viz_tabs[tab_idx]:
                fig_prob = create_probability_distribution_plot(y_test, y_test_proba, n_classes)
                st.plotly_chart(fig_prob, width='stretch')
                
                st.info("""
                **Probability Distribution:**
                - Shows the distribution of predicted probabilities
                - Good separation indicates confident predictions
                - Overlap indicates uncertain predictions
                """)
            tab_idx += 1
            
            with viz_tabs[tab_idx]:
                fig_class = create_class_distribution_plot(y_train, y_test, target_names)
                st.plotly_chart(fig_class, width='stretch')
                
                st.info("""
                **Class Distribution:**
                - Shows the distribution of classes in train and test sets
                - Balanced classes generally lead to better SVM performance
                """)
            tab_idx += 1
            
            if n_classes == 2:
                with viz_tabs[tab_idx]:
                    fig_roc = create_roc_curve_plot(y_test, y_test_proba, n_classes)
                    if fig_roc:
                        st.plotly_chart(fig_roc, width='stretch')
                        
                        st.info("""
                        **ROC Curve:**
                        - Shows trade-off between true positive rate and false positive rate
                        - Area Under Curve (AUC) summarizes overall performance
                        - AUC = 0.5 is random, AUC = 1.0 is perfect
                        """)
                tab_idx += 1
            
            if len(feature_names) == 2:
                with viz_tabs[tab_idx]:
                    try:
                        fig_boundary = create_decision_boundary_plot(
                            X_test, y_test, model, feature_names
                        )
                        if fig_boundary:
                            st.plotly_chart(fig_boundary, width='stretch')
                            
                            st.info("""
                            **Decision Boundary:**
                            - Color gradient shows decision function values across the feature space
                            - Data points are colored by their true class
                            - Gold circles represent support vectors
                            - The boundary line is where the decision function equals zero
                            """)
                        else:
                            st.warning("Decision boundary visualization not available.")
                    except Exception as e:
                        st.error(f"Error creating decision boundary plot: {str(e)}")
                tab_idx += 1
                
                if n_classes == 2:
                    with viz_tabs[tab_idx]:
                        try:
                            fig_margin = create_margin_plot(
                                X_test, y_test, model, feature_names
                            )
                            if fig_margin:
                                st.plotly_chart(fig_margin, width='stretch')
                                
                                st.info("""
                                **Margins:**
                                - Shows the decision function values (distance from hyperplane)
                                - The decision boundary is where the function equals 0
                                - Margins are at -1 and +1 (support vectors lie on margins)
                                - Points between margins are within the margin zone
                                """)
                            else:
                                st.warning("Margin visualization not available.")
                        except Exception as e:
                            st.error(f"Error creating margin plot: {str(e)}")
    
    else:
        st.info("Please select a data source from the sidebar to get started!")


if __name__ == "__main__":
    main()

