import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull

df = pd.read_csv('customers.csv').head(50)
print("Descriping DataFrame \n",df.describe())

features = df[['annual income', 'spending score', 'browsing time', 'frequency purchase']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

oldScore = noOfCluster = 0
for i in range(2,6):
    kmeans = KMeans(n_clusters=i, random_state=1)
    kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features,kmeans.labels_)
    if score > oldScore :
        noOfCluster = i
        oldScore = score


#elbow method
# inertia = []
# for i in range(2,11):
#     kmeans = KMeans(n_clusters=i, random_state=1)
#     kmeans.fit(scaled_features)
#     inertia.append(kmeans.inertia_)

# inertia_differences = np.diff(inertia)
# elbow_point = np.argmin(inertia_differences) + 2
# noOfCluster = elbow_point;

print("No of Cluster suitable for maximize the performace of the data set ",noOfCluster);
kmeans = KMeans(n_clusters=noOfCluster, random_state=1)
df['Cluster'] = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df["PCA1"] = pca_result[:,0]
df["PCA2"] = pca_result[:,1]
pca_center = pca.transform(kmeans.cluster_centers_)

print('\np X-axix \n',df["PCA1"])
print('\np Y-axis \n',df["PCA2"])
print('\np Center X-axis \n',pca_center[:,0])
print('\np Center Y-axis \n',pca_center[1])


plt.figure(figsize=(8, 6))

color = ["Green","Blue","yellow","orange","purple","pink","brown","Red","cyan","Magenta","black"]
for itr in range(noOfCluster):
    print("\n Cluster ",itr);
    print("\n",df.loc[df['Cluster'] == itr])
    plt.scatter(df.loc[df['Cluster'] == itr, 'PCA1'],df.loc[df['Cluster'] == itr, 'PCA2'],c=color[itr],s=200,alpha=0.7,label=("cluster "+str(itr)))
    plt.scatter(pca_center[itr,0],pca_center[itr,1],marker='X',s=300,alpha=0.8,c=color[itr])
    if len(df) >= 3:  # Convex hull requires at least 3 points
        cluster_points = df.loc[df['Cluster'] == itr, ['PCA1','PCA2']].values
        hull = ConvexHull(cluster_points)
        
        # Plot the convex hull boundary
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                     color=color[itr], alpha=0.5)

        # Optionally fill the convex hull with a transparent background color
        plt.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], 
                 color=color[itr], alpha=0.2)


plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.title('K-Means Clustering')
plt.legend()
plt.savefig("cluster.png")

score = silhouette_score(scaled_features,kmeans.labels_)
print("\n Silhouette Score:", score)
