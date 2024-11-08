# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the survey data
# Replace 'survey_data.csv' with the actual file path
data = pd.read_csv('survey_data.csv')

# Preview the data
print("Initial Data Preview:")
print(data.head())

# Data Preprocessing
# Handle missing values
data = data.dropna()

# Map binary variables explicitly
# For example, 'Election' is mapped to 1 (Yes) and 0 (No)
data['Election'] = data['Election'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
label_encoders = {}
# Exclude already encoded 'Election' from categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('Election', errors='ignore')

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Scale numerical variables
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
# Exclude 'Election' from scaling
numerical_columns = numerical_columns.drop('Election', errors='ignore')
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Add 'Election' back to numerical columns for modeling
numerical_columns = numerical_columns.tolist() + ['Election']

# Exploratory Data Analysis
# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data[numerical_columns].corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data[numerical_columns])
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title('PCA of Survey Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Clustering using K-Means
# Determine the optimal number of clusters using the Elbow Method
wcss = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[numerical_columns])
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(K, wcss, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose the optimal number of clusters (e.g., k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data[numerical_columns])

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue=data['KMeans_Cluster'], data=pca_df, palette='viridis')
plt.title('K-Means Clusters of Survey Participants')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Hierarchical Clustering
# Using Agglomerative Clustering and Dendrogram
linked = linkage(data[numerical_columns], method='ward')

# Plot Dendrogram
plt.figure(figsize=(12, 7))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
data['Agglomerative_Cluster'] = agg_cluster.fit_predict(data[numerical_columns])

# Factor Analysis
# Determine the number of factors to extract using Scree Plot
fa = FactorAnalysis()
fa.fit(data[numerical_columns])
eigenvalues, _ = np.linalg.eig(np.cov(data[numerical_columns].T))

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--')
plt.show()

# Select number of factors (e.g., n_factors=5 based on Scree Plot)
n_factors = 5
fa = FactorAnalysis(n_components=n_factors, random_state=42)
fa_components = fa.fit_transform(data[numerical_columns])
fa_df = pd.DataFrame(fa_components, columns=[f'Factor_{i+1}' for i in range(n_factors)])

# Display factor loadings
factor_loadings = pd.DataFrame(fa.components_, columns=data[numerical_columns])
print("Factor Loadings:")
print(factor_loadings.T)

# ANOVA
# Test if there are significant differences in 'Political_Views' across clusters
# Ensure 'Political_Views' and 'KMeans_Cluster' are present
if 'Political_Views' in data.columns:
    anova_data = data[['Political_Views', 'KMeans_Cluster']]
    anova_data['KMeans_Cluster'] = anova_data['KMeans_Cluster'].astype('category')

    model = ols('Political_Views ~ C(KMeans_Cluster)', data=anova_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("ANOVA Table for 'Political_Views' across K-Means Clusters:")
    print(anova_table)
else:
    print("'Political_Views' column not found in data.")

# Logistic Regression
# Predict whether a participant plans to vote in the upcoming election ('Election' variable)
# Define features and target variable
X = data.drop(columns=['Election', 'KMeans_Cluster', 'Agglomerative_Cluster'])
y = data['Election']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Predict on test set
y_pred = log_reg.predict(X_test)

# Evaluation
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Additional Statistical Analysis
# Chi-Squared Test for independence between categorical variables
# Example: Between 'Party_Affiliation' and 'KMeans_Cluster'
if 'Party_Affiliation' in data.columns:
    contingency_table = pd.crosstab(data['Party_Affiliation'], data['KMeans_Cluster'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Squared Test between 'Party_Affiliation' and 'KMeans_Cluster':")
    print(f"Chi2 Statistic: {chi2:.2f}, p-value: {p:.4f}")
else:
    print("'Party_Affiliation' column not found in data.")

# T-Test between clusters for a numerical variable
# Example: 'Political_Views' between Cluster 0 and Cluster 1
if 'Political_Views' in data.columns:
    cluster0 = data[data['KMeans_Cluster'] == 0]['Political_Views']
    cluster1 = data[data['KMeans_Cluster'] == 1]['Political_Views']
    t_stat, p_val = ttest_ind(cluster0, cluster1)
    print(f"T-Test between Cluster 0 and Cluster 1 for 'Political_Views':")
    print(f"T-Statistic: {t_stat:.2f}, p-value: {p_val:.4f}")
else:
    print("'Political_Views' column not found in data.")

# Descriptive statistics by cluster
cluster_summary = data.groupby('KMeans_Cluster').mean()
print("Descriptive Statistics by K-Means Cluster:")
print(cluster_summary)

# Save the processed data with cluster labels
data.to_csv('processed_survey_data.csv', index=False)
print("Processed data saved to 'processed_survey_data.csv'.")