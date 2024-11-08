# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# For NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# For network analysis
import networkx as nx

# For interactive dashboards
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# For performance optimization
from joblib import Parallel, delayed
import multiprocessing

# For ethical considerations (e.g., data anonymization)
from sklearn.utils import shuffle

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Data Ingestion
# -------------------------------

# Replace 'survey_data.csv' with your actual data file path
data_file = 'survey_data.csv'

# Read the survey data into a pandas DataFrame
try:
    df = pd.read_csv(data_file)
    print("Data successfully loaded.")
except FileNotFoundError:
    print(f"File {data_file} not found. Please check the file path.")

# Display the first few rows
print(df.head())

# -------------------------------
# 2. Preprocessing
# -------------------------------

# a. Data Cleaning

# Identify missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values
# For numerical columns, use median imputation
# For categorical columns, use mode imputation

# Define numerical and categorical columns
# Adjust these lists based on your actual data
numerical_cols = ['Age', 'Household_Income', 'Political_Views']
categorical_cols = ['Year', 'Ethnicity', 'Race', 'Party_Affiliation', 
                    'R_Strength', 'D_Strength', 'Independent_Lean', 
                    'Party_Registration', 'Election']

# Impute numerical columns
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Impute categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# b. Data Transformation

# Normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Feature Engineering (example: combine Age and Political_Views)
df['Age_Political_Views'] = df['Age'] * df['Political_Views']

# c. Encoding Categorical Variables

# One-Hot Encoding for nominal categorical variables
df = pd.get_dummies(df, columns=['Year', 'Ethnicity', 'Race', 
                                 'Party_Affiliation', 'R_Strength', 
                                 'D_Strength', 'Independent_Lean', 
                                 'Party_Registration'], drop_first=True)

# Encode the target variable 'Election' as binary
# Assuming 'Yes' is 1 and 'No' is 0
df['Election'] = df['Election'].map({1:1, 2:0})

# d. Encoding Open-Ended Responses

# Assume 'Define' and 'Detect' are the open-ended textual responses
# Preprocess text data

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to open-ended responses
for col in ['Define', 'Detect']:
    df[col] = df[col].astype(str).apply(preprocess_text)

# Ethical Considerations: Data Anonymization
# If there are any unique identifiers, remove or anonymize them
# Assuming 'Respondent_ID' is a unique identifier (adjust as necessary)
if 'Respondent_ID' in df.columns:
    df = df.drop('Respondent_ID', axis=1)

print("\nData after preprocessing:")
print(df.head())

# -------------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------------

# a. Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# b. Visualization

# Univariate Analysis - Histogram for Age
plt.figure(figsize=(8,6))
sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age (Standardized)')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis - Scatter plot between Age and Political_Views
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Political_Views', data=df, hue='Election')
plt.title('Age vs. Political Views')
plt.xlabel('Age (Standardized)')
plt.ylabel('Political Views (Standardized)')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# c. Correlation Analysis

# Pearson correlation for numerical variables
pearson_corr = df[numerical_cols + ['Age_Political_Views']].corr()
print("\nPearson Correlation:")
print(pearson_corr)

# Cramér's V for categorical variables
# Function to compute Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = sns.utils.contingency.expected_freq(confusion_matrix)
    from scipy.stats import chi2_contingency
    chi2, p, dof, ex = chi2_contingency(confusion_matrix, correction=False)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1))))

# Example: Cramér's V between 'Party_Affiliation_Democrat' and 'Election'
if 'Party_Affiliation_Democrat' in df.columns and 'Election' in df.columns:
    cramers = cramers_v(df['Party_Affiliation_Democrat'], df['Election'])
    print(f"\nCramér's V between Party_Affiliation_Democrat and Election: {cramers:.3f}")

# -------------------------------
# 4. Dimensionality Reduction (PCA)
# -------------------------------

# Select features for PCA
features = df.drop(['Election', 'Define', 'Detect'], axis=1)

# Standardize the features
scaler_pca = StandardScaler()
features_scaled = scaler_pca.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)  # Adjust the number of components as needed
principal_components = pca.fit_transform(features_scaled)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df = pd.concat([pca_df, df['Election']], axis=1)

# Visualize PCA results
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Election', data=pca_df, palette='viridis')
plt.title('PCA - Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# -------------------------------
# 5. Clustering (K-Means)
# -------------------------------

# Determine optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.show()

# From the Elbow plot, choose k (e.g., k=3)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('Clusters based on PCA Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# -------------------------------
# 6. Predictive Modeling (Random Forest with Hyperparameter Tuning)
# -------------------------------

# Define target and features
X = df.drop(['Election', 'Define', 'Detect'], axis=1)
y = df['Election']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV
print("\nStarting Grid Search for Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print("\nBest Hyperparameters:", best_params)

# Best estimator
best_rf = grid_search.best_estimator_

# Predict on test set using the best model
y_pred = best_rf.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classifier Report (After Hyperparameter Tuning):")
print(classification_report(y_test, y_pred))

# Feature Importance from the tuned model
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10,8))
sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10], palette='viridis')
plt.title('Top 10 Feature Importances (After Tuning)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# -------------------------------
# 7. Semantic Analysis (Sentiment Analysis)
# -------------------------------

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def get_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

# Apply sentiment analysis to open-ended responses
df['Define_Sentiment'] = df['Define'].apply(get_sentiment)
df['Detect_Sentiment'] = df['Detect'].apply(get_sentiment)

# Visualize sentiment distributions
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['Define_Sentiment'], kde=True, color='skyblue')
plt.title('Sentiment Distribution - Define')
plt.xlabel('Sentiment Score')

plt.subplot(1,2,2)
sns.histplot(df['Detect_Sentiment'], kde=True, color='salmon')
plt.title('Sentiment Distribution - Detect')
plt.xlabel('Sentiment Score')

plt.tight_layout()
plt.show()

# Correlate sentiment with voting intention
plt.figure(figsize=(10,6))
sns.boxplot(x='Election', y='Define_Sentiment', data=df, palette='Set2')
plt.title('Define Sentiment by Voting Intention')
plt.xlabel('Election (0=No, 1=Yes)')
plt.ylabel('Sentiment Score')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Election', y='Detect_Sentiment', data=df, palette='Set2')
plt.title('Detect Sentiment by Voting Intention')
plt.xlabel('Election (0=No, 1=Yes)')
plt.ylabel('Sentiment Score')
plt.show()

# -------------------------------
# 8. Network Construction (Correlation Network)
# -------------------------------

# Compute correlation matrix
corr_matrix = df.corr()

# Threshold for significant correlations
threshold = 0.3

# Create a graph from the correlation matrix
G = nx.Graph()

# Add nodes
for col in corr_matrix.columns:
    G.add_node(col)

# Add edges with correlation above threshold
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i,j]) > threshold:
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i,j])

# Plot the network
plt.figure(figsize=(15,15))
pos = nx.spring_layout(G, k=0.15)
edges = G.edges(data=True)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
plt.title('Correlation Network of Survey Questions')
plt.axis('off')
plt.show()

# -------------------------------
# 9. Visualization (Interactive Dashboards)
# -------------------------------

# Enhanced Dash Application with more plots, filters, and interactive components

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

# Prepare data for interactive components
# Example: Include Cluster and Sentiment in PCA DataFrame
pca_df = pd.concat([pca_df, df['Cluster'], df['Define_Sentiment'], df['Detect_Sentiment']], axis=1)

# Create interactive figures
# PCA Scatter Plot with filters
pca_fig = px.scatter(
    pca_df, x='PC1', y='PC2',
    color='Election',
    symbol='Cluster',
    hover_data=['Election', 'Cluster', 'Define_Sentiment', 'Detect_Sentiment'],
    title='PCA Scatter Plot: Election Intention and Clusters',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
)

# Feature Importance Bar Chart
feature_imp_fig = px.bar(
    feature_importances[:10].sort_values(),
    x=feature_importances[:10].sort_values(),
    y=feature_importances.index[:10].sort_values(),
    orientation='h',
    title='Top 10 Feature Importances (After Tuning)',
    labels={'x': 'Importance Score', 'y': 'Features'}
)

# Sentiment Distribution Histogram
sentiment_fig = go.Figure()
sentiment_fig.add_trace(go.Histogram(
    x=df['Define_Sentiment'],
    name='Define Sentiment',
    opacity=0.75
))
sentiment_fig.add_trace(go.Histogram(
    x=df['Detect_Sentiment'],
    name='Detect Sentiment',
    opacity=0.75
))
sentiment_fig.update_layout(
    title='Sentiment Distribution',
    xaxis_title='Sentiment Score',
    yaxis_title='Count',
    barmode='overlay',
    template='plotly_white'
)
sentiment_fig.update_traces(opacity=0.6)

# Correlation Heatmap
corr_fig = px.imshow(
    corr_matrix,
    text_auto=False,
    aspect='auto',
    color_continuous_scale='RdBu_r',
    title='Correlation Heatmap'
)

# Cluster Distribution Pie Chart
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
pie_fig = px.pie(
    cluster_counts,
    names='Cluster',
    values='Count',
    title='Distribution of Clusters'
)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("PSY 492 Election Experience Survey Analysis Dashboard", style={'textAlign': 'center'}),
    
    # Filters Section
    html.Div([
        html.H2("Filters", style={'marginTop': '40px'}),
        
        html.Label("Select Party Affiliation:"),
        dcc.Dropdown(
            id='party-filter',
            options=[{'label': col, 'value': col} for col in df.columns if 'Party_Affiliation' in col],
            multi=True,
            value=[],
            placeholder="Select Party Affiliations"
        ),
        
        html.Label("Select Household Income Range:", style={'marginTop': '20px'}),
        dcc.RangeSlider(
            id='income-filter',
            min=df['Household_Income'].min(),
            max=df['Household_Income'].max(),
            step=1,
            marks={int(i): str(i) for i in np.linspace(df['Household_Income'].min(), df['Household_Income'].max(), num=5)},
            value=[df['Household_Income'].min(), df['Household_Income'].max()]
        ),
        
        html.Label("Select Age Range (Standardized):", style={'marginTop': '20px'}),
        dcc.RangeSlider(
            id='age-filter',
            min=df['Age'].min(),
            max=df['Age'].max(),
            step=0.1,
            marks={-2: '-2', 0: '0', 2: '2'},
            value=[df['Age'].min(), df['Age'].max()]
        ),
    ], style={'width': '80%', 'margin': 'auto'}),
    
    # Interactive Plots Section
    html.Div([
        html.H2("Interactive Plots", style={'marginTop': '40px'}),
        
        dcc.Graph(id='pca-scatter'),
        dcc.Graph(id='feature-importance'),
        dcc.Graph(id='sentiment-distribution'),
        dcc.Graph(id='correlation-heatmap'),
        dcc.Graph(id='cluster-pie-chart'),
    ], style={'width': '90%', 'margin': 'auto'}),
    
    # Additional Visualizations
    html.Div([
        html.H2("Additional Visualizations", style={'marginTop': '40px'}),
        
        # Correlation Heatmap
        dcc.Graph(
            id='correlation-heatmap',
            figure=corr_fig
        ),
        
        # Cluster Distribution Pie Chart
        dcc.Graph(
            id='cluster-pie-chart',
            figure=pie_fig
        ),
    ], style={'width': '90%', 'margin': 'auto'}),
    
])

# Callback to update plots based on filters
@app.callback(
    [
        Output('pca-scatter', 'figure'),
        Output('feature-importance', 'figure'),
        Output('sentiment-distribution', 'figure'),
        Output('correlation-heatmap', 'figure'),
        Output('cluster-pie-chart', 'figure'),
    ],
    [
        Input('party-filter', 'value'),
        Input('income-filter', 'value'),
        Input('age-filter', 'value')
    ]
)
def update_plots(selected_parties, income_range, age_range):
    # Filter data based on selections
    filtered_df = df.copy()
    
    if selected_parties:
        for party in selected_parties:
            if party in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[party] == 1]
    
    filtered_df = filtered_df[
        (filtered_df['Household_Income'] >= income_range[0]) &
        (filtered_df['Household_Income'] <= income_range[1]) &
        (filtered_df['Age'] >= age_range[0]) &
        (filtered_df['Age'] <= age_range[1])
    ]
    
    # Recompute PCA
    features_filtered = filtered_df.drop(['Election', 'Define', 'Detect'], axis=1)
    features_scaled_filtered = scaler_pca.transform(features_filtered)
    principal_components_filtered = pca.transform(features_scaled_filtered)
    pca_df_filtered = pd.DataFrame(data=principal_components_filtered, columns=['PC1', 'PC2'])
    pca_df_filtered = pd.concat([pca_df_filtered, filtered_df['Election'], filtered_df['Cluster'],
                                 filtered_df['Define_Sentiment'], filtered_df['Detect_Sentiment']], axis=1)
    
    # Update PCA Scatter Plot
    pca_fig_updated = px.scatter(
        pca_df_filtered, x='PC1', y='PC2',
        color='Election',
        symbol='Cluster',
        hover_data=['Election', 'Cluster', 'Define_Sentiment', 'Detect_Sentiment'],
        title='PCA Scatter Plot: Election Intention and Clusters',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
    )
    
    # Update Feature Importance (static as feature importance doesn't change with data)
    
    # Update Sentiment Distribution
    sentiment_fig_updated = go.Figure()
    sentiment_fig_updated.add_trace(go.Histogram(
        x=filtered_df['Define_Sentiment'],
        name='Define Sentiment',
        opacity=0.75
    ))
    sentiment_fig_updated.add_trace(go.Histogram(
        x=filtered_df['Detect_Sentiment'],
        name='Detect Sentiment',
        opacity=0.75
    ))
    sentiment_fig_updated.update_layout(
        title='Sentiment Distribution',
        xaxis_title='Sentiment Score',
        yaxis_title='Count',
        barmode='overlay',
        template='plotly_white'
    )
    sentiment_fig_updated.update_traces(opacity=0.6)
    
    # Update Correlation Heatmap
    if not filtered_df.empty:
        corr_matrix_filtered = filtered_df.corr()
        corr_fig_updated = px.imshow(
            corr_matrix_filtered,
            text_auto=False,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Correlation Heatmap (Filtered Data)'
        )
    else:
        # Empty figure
        corr_fig_updated = go.Figure()
        corr_fig_updated.update_layout(title='Correlation Heatmap (Filtered Data - No Data)')
    
    # Update Cluster Distribution Pie Chart
    cluster_counts_filtered = filtered_df['Cluster'].value_counts().reset_index()
    cluster_counts_filtered.columns = ['Cluster', 'Count']
    pie_fig_updated = px.pie(
        cluster_counts_filtered,
        names='Cluster',
        values='Count',
        title='Distribution of Clusters (Filtered Data)'
    )
    
    return pca_fig_updated, feature_imp_fig, sentiment_fig_updated, corr_fig_updated, pie_fig_updated

# Run the Dash app
# To launch the dashboard, uncomment the following line and run the script
# app.run_server(debug=True)

# -------------------------------
# 10. Validation
# -------------------------------

# a. Model Validation using Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)

print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Standard Deviation of CV Accuracy:", cv_scores.std())

# b. Statistical Significance

# Permutation Importance
perm_importance = permutation_importance(best_rf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

# Create a series with feature importances
perm_importances = pd.Series(perm_importance.importances_mean, index=X.columns).sort_values(ascending=False)

# Plot permutation importances
plt.figure(figsize=(10,8))
sns.barplot(x=perm_importances[:10], y=perm_importances.index[:10], palette='magma')
plt.title('Permutation Feature Importances (After Tuning)')
plt.xlabel('Mean Importance Score')
plt.ylabel('Features')
plt.show()

# -------------------------------
# Ethical Considerations and Documentation
# -------------------------------

"""
Ethical Considerations:
- **Data Privacy:** Ensure that all personal and sensitive information is anonymized. In this script, 'Respondent_ID' is dropped to prevent identification.
- **Data Handling:** Securely store and handle data to prevent unauthorized access.
- **Bias Mitigation:** During preprocessing and modeling, be cautious of introducing or perpetuating biases. Ensure that the model does not unfairly discriminate against any group.
- **Transparency:** Document all preprocessing steps, feature engineering, and modeling decisions to maintain transparency and reproducibility.

Documentation and Reporting:
- **Jupyter Notebook:** It is recommended to create a Jupyter Notebook that includes narrative explanations, visualizations, and code snippets to provide a comprehensive report of the analysis.
- **Code Comments:** The script includes detailed comments explaining each step. Continue to maintain clear and concise comments for future reference and for other users.
- **Version Control:** Use version control systems like Git to track changes and collaborate with others.
- **Reproducibility:** Ensure that all steps can be reproduced by setting random seeds and documenting library versions.

Example Structure for a Jupyter Notebook:
1. **Introduction:** Overview of the survey and objectives of the analysis.
2. **Data Ingestion:** Loading and initial inspection of the data.
3. **Preprocessing:** Detailed steps of data cleaning, transformation, and encoding with explanations.
4. **Exploratory Data Analysis (EDA):** Visualizations and insights derived from the data.
5. **Dimensionality Reduction:** Explanation of PCA and its findings.
6. **Clustering:** Details of clustering methodology and interpretation of clusters.
7. **Predictive Modeling:** Building, tuning, and evaluating the Random Forest model.
8. **Semantic Analysis:** Performing and interpreting sentiment analysis on open-ended responses.
9. **Network Analysis:** Construction and interpretation of the correlation network.
10. **Interactive Dashboards:** Description and screenshots of the Dash application.
11. **Validation:** Cross-validation results and statistical significance of findings.
12. **Ethical Considerations:** Discussion on data privacy, bias, and ethical handling of data.
13. **Conclusion:** Summary of key findings and potential recommendations.
14. **References:** Cite any sources or libraries used.

"""

# -------------------------------
# End of Enhanced Script
# -------------------------------
