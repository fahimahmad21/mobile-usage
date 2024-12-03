import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_excel('usage.xlsx')

df.rename(index=str, columns={
    'Total_App_Usage_Hours': 'App_Usage',
    'Daily_Screen_Time_Hours': 'Daily_Screen_Time',
    'Social_Media_Usage_Hours': 'Social_Media',
    'Productivity_App_Usage_Hours': 'Productivity_App',
    'Gaming_App_Usage_Hours': 'Gaming_App'  
}, inplace=True)


X = df.drop(['User_ID', 'Gender', 'Location'], axis=1)

st.header("dataset content")
st.write(X)

# Show Elbow
st.header("Elbow Curve")
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

# Membuat plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)

# Menambahkan judul dan label
ax.set_title('Find Elbow Point')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')

# Menambahkan anotasi untuk titik elbow pertama (misalnya klaster 3)
ax.annotate('Possible elbow point', 
            xy=(3, clusters[2]),  # Titik koordinat untuk elbow pertama (klaster 3)
            xytext=(3, clusters[2] + 50000),  # Posisi teks
            xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

# Menambahkan anotasi untuk titik elbow kedua (misalnya klaster 5)
ax.annotate('Possible elbow point', 
            xy=(5, clusters[4]),  # Titik koordinat untuk elbow kedua (klaster 5)
            xytext=(5, clusters[4] + 50000),  # Posisi teks
            xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

# Menampilkan plot
st.pyplot(fig)

# Melakukan clustering
st.sidebar.subheader("Cluster number")
clust = st.sidebar.slider("Choice cluster :", 2,10,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    X_reduced = X[['Age','App_Usage', 'Daily_Screen_Time', 'Number_of_Apps_Used', 'Social_Media', 'Productivity_App', 'Gaming_App']]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_reduced)

    # Menambahkan hasil PCA ke DataFrame untuk plotting
    X['PCA_1'] = X_pca[:, 0]
    X['PCA_2'] = X_pca[:, 1]

    # Membuat scatterplot dengan dua dimensi yang baru (PCA_1 dan PCA_2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='PCA_1', y='PCA_2', hue='Labels', size='Labels', data=X, palette=sns.color_palette('hls', n_colors=len(X['Labels'].unique())))

    # Menambahkan anotasi untuk setiap label
    for label in X['Labels'].unique():
        label_data = X[X['Labels'] == label]
        plt.annotate(f'Cluster {label}',
                    (label_data['PCA_1'].mean(), label_data['PCA_2'].mean()),
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=12, weight='bold',
                    color='black')

    plt.title('Clustering Visualization Using PCA')
    st.header('Cluster Plot')
    st.pyplot(fig)
    st.write(X)

k_means(clust)