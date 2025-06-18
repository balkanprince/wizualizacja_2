
import pandas as pd
import numpy as np
import os
import re
import zipfile
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError:
    print("Biblioteka 'nltk' nie jest zainstalowana. Zainstaluj ją za pomocą: pip install nltk")
    exit()
from kaggle.api.kaggle_api_extended import KaggleApi
print("Kaggle zaimportowane poprawnie")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Pobierz zasoby NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Błąd podczas pobierania zasobów NLTK: {e}")
    exit()

# Funkcja do czyszczenia tekstu
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Pobieranie danych z Kaggle
def download_cord19_data(base_path, dataset='allen-institute-for-ai/CORD-19-research-challenge'):
    data_dir = os.path.join(base_path, 'data')
    metadata_file = os.path.join(data_dir, 'metadata.csv')
    
    if os.path.exists(metadata_file):
        print(f"Plik {metadata_file} już istnieje. Pomijam pobieranie.")
        return metadata_file
    
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        api = KaggleApi()
        api.authenticate()
        print("Pobieranie danych CORD-19 z Kaggle...")
        api.dataset_download_files(dataset, path=data_dir, unzip=False)
        
        zip_path = os.path.join(data_dir, 'CORD-19-research-challenge.zip')
        if os.path.exists(zip_path):
            print(f"Rozpakowywanie pliku {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract('metadata.csv', data_dir)
            os.remove(zip_path)  # Usuń plik ZIP po rozpakowaniu
            print(f"Plik {metadata_file} został pobrany i rozpakowany.")
            return metadata_file
        else:
            raise FileNotFoundError(f"Plik ZIP {zip_path} nie został pobrany.")
    except Exception as e:
        print(f"Błąd podczas pobierania danych z Kaggle: {e}")
        exit()

# Wczytanie danych CORD-19
def load_cord19_data(metadata_file, max_files=1000):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Plik {metadata_file} nie istnieje!")
    
    print(f"Wczytywanie pliku: {metadata_file}")
    df = pd.read_csv(metadata_file, low_memory=False)
    print(f"Wczytano {len(df)} wierszy z pliku metadata.csv")
    
    df = df.dropna(subset=['abstract']).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("Brak abstraktów w danych. Sprawdź, czy kolumna 'abstract' w pliku metadata.csv zawiera dane.")
    print(f"Po odfiltrowaniu wierszy bez abstraktów: {len(df)} wierszy")
    
    df = df.head(max_files)
    df['abstract_clean'] = df['abstract'].apply(preprocess_text)
    return df

# Ścieżka do folderu projektu
base_path = "C:/Users/mpiesio/Desktop/KODILLA/wizualizacja_2"
try:
    metadata_file = download_cord19_data(base_path)
    df = load_cord19_data(metadata_file, max_files=1000)
    print(f"Wczytano {len(df)} artykułów z abstraktami.")
except Exception as e:
    print(e)
    exit()

# Wektoryzacja TF-IDF
try:
    vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.8)
    X_tfidf = vectorizer.fit_transform(df['abstract_clean']).toarray()
    print(f"Wektoryzacja TF-IDF zakończona. Rozmiar macierzy: {X_tfidf.shape}")
except Exception as e:
    print(f"Błąd podczas wektoryzacji TF-IDF: {e}")
    exit()

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_tfidf)

# Redukcja wymiarowości do 2D dla wizualizacji
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Redukcja wymiarowości PCA zakończona. Rozmiar danych: {X_pca.shape}")

# Dobór liczby klastrów dla KMeans
inertias = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Wykres metody łokcia
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, 'bx-')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('SSE')
plt.title('Metoda łokcia dla KMeans')
plt.show()

# Wykres silhouette score
plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score dla KMeans')
plt.show()

# Klasteryzacja: KMeans
n_clusters = 5  # Załóżmy 5 klastrów po analizie
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Wizualizacja KMeans
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('KMeans Clustering (PCA 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Klasteryzacja: DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Wizualizacja DBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', s=50)
plt.title('DBSCAN Clustering (PCA 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Klasteryzacja: Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
gmm_labels = gmm.fit_predict(X_scaled)

# Wizualizacja GMM
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title('Gaussian Mixture Model Clustering (PCA 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Analiza klastrów
for cluster in range(n_clusters):
    print(f"\nKlaster {cluster} (KMeans):")
    cluster_indices = np.where(kmeans_labels == cluster)[0]
    for idx in cluster_indices[:3]:
        title = df.iloc[idx]['title']
        print(f"- {title if pd.notna(title) else 'Brak tytułu'}")

# Zapisz wyniki
df['kmeans_cluster'] = kmeans_labels
df['dbscan_cluster'] = dbscan_labels
df['gmm_cluster'] = gmm_labels
df[['title', 'abstract', 'kmeans_cluster', 'dbscan_cluster', 'gmm_cluster']].to_csv('cord19_clustering_results.csv', index=False)
print("Wyniki zapisano do 'cord19_clustering_results.csv'.") 

