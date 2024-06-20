import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import numpy as np
import time

def load_data(file_path):
    """
    Veri setini yükleyen fonksiyon.
    """
    data = pd.read_csv(file_path, dtype={'NObeyesdad': 'category'})
    return data

def preprocess_data(obesity):
    """
    Veri setini ön işleyen fonksiyon.
    """
    obesity = obesity.dropna()
    y = obesity['NObeyesdad']
    X = obesity.drop(columns=['NObeyesdad'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Farklı sınıflandırma modellerini eğiten ve değerlendiren fonksiyon.
    """
    models = [
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=500, solver='liblinear', random_state=42)),
    ]

    kappa_scores = []
    confusion_matrices = {}
    execution_times = {}

    for name, model in models:
        # Eğitim süresi ölçümü
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Test süresi ölçümü
        start_time = time.time()
        predictions = model.predict(X_test)
        test_time = time.time() - start_time

        # 10 kat çaprazlama ile modeli değerlendir
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        kappa_scores.append((name, scores.mean()))

        # Confusion matrix hesapla
        cm = confusion_matrix(y_test, predictions)
        confusion_matrices[name] = cm

        # Eğitim ve test sürelerini kaydet
        execution_times[name] = (training_time, test_time)

    return kappa_scores, confusion_matrices, models, execution_times

    for name, model in models:
        # Modeli eğit
        model.fit(X_train, y_train)

        # 10 kat çaprazlama ile modeli değerlendir
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        kappa_scores.append((name, scores.mean()))

        # Tahminleri al
        predictions = model.predict(X_test)

        # Confusion matrix hesapla
        cm = confusion_matrix(y_test, predictions)
        confusion_matrices[name] = cm

    return kappa_scores, confusion_matrices, models, execution_times


def plot_confusion_matrices(confusion_matrices):
    """
    Confusion matrislerini görselleştiren fonksiyon.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for ax, (name, cm) in zip(axes.flatten(), confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

    plt.show()

def plot_obesity_classes(obesity):
    plt.figure(figsize=(8, 5))
    obesity['NObeyesdad'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Obezite Sınıfları ve Dağılımları')
    plt.xlabel('Obezite Sınıfı')
    plt.ylabel('Frekans')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_gender_vs_obesity(obesity):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=obesity, x='Gender', hue='NObeyesdad', palette='Set2')
    plt.title('Cinsiyet ve Obezite İlişkisi')
    plt.xlabel('Cinsiyet')
    plt.ylabel('Frekans')
    plt.legend(title='Obezite Sınıfı')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_physical_activity_level(obesity):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=obesity, x='Gender', y='FAF')
    plt.title('Fiziksel Aktivite Sıklığı')
    plt.xlabel('Gender')
    plt.ylabel('Fiziksel Aktivite Sıklığı')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_dietary_habits(obesity):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=obesity, x='Age', y='FCVC')
    plt.title('Yaşa göre sebze tüketim sıklığı')
    plt.xlabel('Age')
    plt.ylabel('Yaşa göre sebze tüketim sıklığı')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_heatmap(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='viridis', fmt=".2f")
    plt.title('Özellikler Arasındaki Korelasyon Isı Haritası')
    plt.show()

def main():
    """
    Ana işlem fonksiyonu.
    """
    # Sabit veri seti yolu
    file_path = r"C:\Users\dagha\OneDrive\Masaüstü\21110131074_Halil Rodi DAĞ\_Kaynak Kod\yzProje\obesity.csv"
    obesity = load_data(file_path)

    # Özelliklerin arasındaki ilişkiyi gösteren ısı haritasını oluştur
    plot_heatmap(obesity)

    # Veri setinin ön işlenmesi
    X_train, X_test, y_train, y_test = preprocess_data(obesity)

    # Modellerin eğitilmesi ve değerlendirilmesi
    kappa_scores, confusion_matrices, models, execution_times = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Kappa skorlarının yazdırılması
    print("Kappa Scores:")
    for name, score in kappa_scores:
        print(f"{name}: {score}")

    # Eğitim ve test sürelerinin yazdırılması
    print("\nEğitim ve Test Süreleri:")
    for name, times in execution_times.items():
        print(f"{name}: Eğitim Süresi = {times[0]:.4f} saniye, Test Süresi = {times[1]:.4f} saniye")

    # Confusion matrislerinin görselleştirilmesi
    plot_confusion_matrices(confusion_matrices)

    # Obezite ile ilgili tabloların gösterilmesi
    plot_obesity_classes(obesity)
    plot_gender_vs_obesity(obesity)
    plot_physical_activity_level(obesity)
    plot_dietary_habits(obesity)

    # Kullanıcı verisini alma ve tahmin etme
    veri = input("Lütfen veri setinden aldığınız veriyi, son rakamı dahil etmeden virgülle ayırarak girin: ")

    # Virgüllerle ayrılmış veriyi bir listeye dönüştür
    veri_listesi = veri.split(',')

    # Veriyi bir veri çerçevesine dönüştürmek için bir sözlük oluştur
    # Modelde kullanılan sütun isimlerini al
    model_columns = X_train.columns

    # Girilen verinin sütun sayısını kontrol et
    if len(veri_listesi) != len(model_columns):
        print(f"Hata: Beklenen {len(model_columns)} sütun, ancak {len(veri_listesi)} sütun girildi.")
        return

    # Veriyi bir veri çerçevesine dönüştürmek için bir sözlük oluştur
    user_data = {}
    for i, deger in enumerate(veri_listesi):
        user_data[model_columns[i]] = [deger]

    user_data_df = pd.DataFrame(user_data)

    # Tahmin yap
    print("\nTahmin Sonuçları:")
    for name, model in models:
        model.fit(X_train, y_train)  # Modeli eğit
        prediction = model.predict(user_data_df)  # Tahmin yap
        print(f"{name}: {prediction}")

if __name__ == "__main__":
    main()