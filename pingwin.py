import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import f1_score

# Wczytanie danych
file_path = r"C:\Users\mpiesio\Desktop\KODILLA\wizualizacja_2\penguins.csv"
df = pd.read_csv(file_path)

# Podstawowe informacje o danych
print("Informacje o danych:")
print(df.info())
print("\nPierwsze 5 wierszy:")
print(df.head())
print("\nBraki danych:")
print(df.isnull().sum())

# Rozkład gatunków
print("\nRozkład gatunków:")
print(df['Species'].value_counts())  # Poprawiono 'species' na 'Species'

# Wizualizacja cech numerycznych
numerical_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']  # Poprawiono nazwy kolumn
df[numerical_cols].hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Wizualizacja cech kategorycznych
# Uwaga: Twój plik nie zawiera kolumn 'island' ani 'sex', więc pomijamy te wizualizacje
# Jeśli masz dodatkowe dane z tymi kolumnami, dodaj je do pliku lub dostosuj kod

# Obsługa braków danych
df['CulmenLength'].fillna(df['CulmenLength'].mean(), inplace=True)
df['CulmenDepth'].fillna(df['CulmenDepth'].mean(), inplace=True)
df['FlipperLength'].fillna(df['FlipperLength'].mean(), inplace=True)
df['BodyMass'].fillna(df['BodyMass'].mean(), inplace=True)

print("\nBraki danych po imputacji:")
print(df.isnull().sum())

# Przygotowanie danych do klasyfikacji
X = df.drop('Species', axis=1)  # Cechy
y = df['Species']  # Zmienna docelowa

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=0, stratify=y)
print("\nKształt zbioru treningowego:", X_train.shape)
print("Kształt zbioru testowego:", X_test.shape)

# Wybór dwóch cech do wizualizacji
X_train_2d = X_train[['CulmenLength', 'CulmenDepth']]
X_test_2d = X_test[['CulmenLength', 'CulmenDepth']]

# Funkcja do oceny modeli
def calculate_metrics(model, model_name, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')
    print(f"{model_name} - F1 trening (makro): {train_f1:.4f}, F1 test (makro): {test_f1:.4f}")
    return train_f1, test_f1

# Funkcja do wizualizacji powierzchni klasyfikacji
def plot_classification_surface(X_plot, y_plot, trained_model):
    x_min, x_max = X_plot.iloc[:, 0].min() - 1, X_plot.iloc[:, 0].max() + 1
    y_min, y_max = X_plot.iloc[:, 1].min() - 1, X_plot.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = trained_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y_plot, alpha=0.8)
    plt.title(f"Powierzchnia klasyfikacji - {trained_model.__class__.__name__}")
    plt.xlabel("Długość dzioba (standaryzowana)")
    plt.ylabel("Głębokość dzioba (standaryzowana)")
    plt.show()

# KNN - GridSearchCV
knn_params = {'n_neighbors': [3, 5, 10, 15], 'metric': ['euclidean', 'manhattan']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, scoring='f1_macro', cv=5, n_jobs=-1)
knn_grid.fit(X_train, y_train)
print("KNN - Najlepsze parametry:", knn_grid.best_params_)

# Trening na pełnym zestawie cech
knn_model = knn_grid.best_estimator_
knn_train_f1, knn_test_f1 = calculate_metrics(knn_model, "KNN", X_train, y_train, X_test, y_test)

# Trening na zestawie 2D do wizualizacji
knn_model_2d = KNeighborsClassifier(**knn_grid.best_params_)
knn_model_2d.fit(X_train_2d, y_train)
plot_classification_surface(X_train_2d, y_train, knn_model_2d)