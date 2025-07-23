import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load your customer data
df = pd.read_csv('customers.csv').head(25)  # Ensure this file has the correct column names

# Select relevant features
features = df[['annual income', 'spending score', 'browsing time', 'frequency purchase']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform KMeans clustering on all 4 features
noOfCluster = 2
kmeans = KMeans(n_clusters=noOfCluster, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Reduce dimensions to 2D using PCA for plotting
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Store PCA components for plotting
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Apply PCA to centroids for visualization
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Plot the clusters in PCA space
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['PC1'], df['PC2'],
                      c=df['Cluster'], cmap='tab10', s=100, edgecolors='k', alpha=0.7)

# Plot centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            marker='X', s=300, c='black', label='Centroids')

# Annotate customer IDs (optional)
for i in range(len(df)):
    plt.text(df['PC1'][i] + 0.1, df['PC2'][i] + 0.1, str(df['customer_id'][i]), fontsize=8)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clusters Visualized in PCA Space (4D → 2D)')
plt.legend()
plt.grid(True)
plt.show()

score = silhouette_score(scaled_features,kmeans.labels_)
print("Silhouette Score:", score)

# Create dictionary of cluster members
mycluster = {}
for i in range(len(df)):
    key = "c" + str(df['Cluster'][i])
    if key not in mycluster:
        mycluster[key] = []
    mycluster[key].append(df['customer_id'][i])

# Print results
print("Customer groups by cluster:")
print(mycluster)

print("\nShape of original KMeans cluster centers (scaled):", kmeans.cluster_centers_.shape)
print("Cluster centers (original scale):\n", scaler.inverse_transform(kmeans.cluster_centers_))