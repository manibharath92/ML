
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df = pd.read_csv('customers.csv')# Assume it has the relevant columns
print(df.describe())

features = df[['annual income', 'spending score', 'browsing time', 'frequency purchase']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

noOfCluster=2
kmeans = KMeans(n_clusters=noOfCluster, random_state=42)
#kmeans = KMeans(n_clusters=KMeans(n_clusters=noOfCluster)
df['Cluster'] = kmeans.fit_predict(scaled_features)


# Plotting clusters using original features (annual income vs spending score)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['annual income'], df['spending score'],
                      c=df['Cluster'], cmap='tab10', s=200, edgecolors='k', alpha=0.7)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Clusters with Centroids')
#plt.grid(True)

# Annotate customer_id
for i in range(len(df)):
    plt.text(df['annual income'][i] + 1000, df['spending score'][i] + 1, df['customer_id'][i], fontsize=9)

# Inverse transform centroids to original scale for plotting
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot centroids in black with marker 'X'
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=300, c='black', label='Centroids')

#plt.scatter(50000, 50,c='red', s=250, marker='*', label='New Customer')

plt.legend()
#plt.show()
plt.savefig("cluster.png")	

score = silhouette_score(scaled_features,kmeans.labels_)
print("Silhouette Score:", score)

mycluster = {}
for i in range(len(df)):
    key = "c" + str(df['Cluster'][i])
    if key not in mycluster:
        mycluster[key] = []
    mycluster[key].append(df['customer_id'][i])

print(mycluster)
print("Shape of cluster centers:", kmeans.cluster_centers_)
print("centeoid:", centroids)