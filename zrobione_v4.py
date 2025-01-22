import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Ścieżka do pliku CSV
sciezka = r"C:\program_przejsciowa\filtered_distances.csv"

# Wczytaj dane z pliku CSV
df = pd.read_csv(sciezka)
odleglosci = df['odleglosc_w_metrach'].values

# Funkcja przypisująca etykiety klas na podstawie wartości odległości
def przypisz_klase(odleglosc):
    if odleglosc < 10.66:
        return 0  # niebezpieczna
    elif odleglosc < 18.33:
        return 1  # ryzyko
    else:
        return 2  # bezpieczna

# Przypisanie etykiet do całego zbioru danych
etykiety = np.array([przypisz_klase(d) for d in odleglosci])
X = odleglosci.reshape(-1, 1)
y = to_categorical(etykiety, num_classes=3)

# Definicja klas dla raportu i wizualizacji
classes = ['Niebezpieczna', 'Ryzyko', 'Bezpieczna']

# Konfiguracja k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

wszystkie_dokladnosci = []
ostatni_history = None
ostatni_model = None
ostatnie_y_test = None
ostatnie_y_pred = None
ostatnie_y_test_classes = None

fold = 1
for train_index, val_index in kf.split(X):
    print(f"\n--- Fold {fold} ---")
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    # Budowa modelu dla bieżącego folda
    model = Sequential([
        Dense(32, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Trenowanie modelu na bieżącym foldzie
    history = model.fit(X_train_fold, y_train_fold,
                        epochs=150,
                        batch_size=128,
                        validation_data=(X_val_fold, y_val_fold),
                        verbose=0)
    
    # Ocena modelu na zbiorze walidacyjnym folda
    wynik = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Dokładność dla fold {fold}: {wynik[1]:.4f}")
    wszystkie_dokladnosci.append(wynik[1])
    
    # Zapisz ostatni model oraz dane do dalszych analiz (dla ostatniego folda)
    ostatni_history = history
    ostatni_model = model
    ostatnie_y_test = y_val_fold
    fold += 1

# Średnia dokładność po k-fold cross-validation
srednia_dokladnosc = np.mean(wszystkie_dokladnosci)
print(f"\nŚrednia dokładność po {k}-fold cross-validation: {srednia_dokladnosc:.4f}")

# Dla ostatniego folda: wykonaj dodatkowe analizy

# Predykcja na zbiorze walidacyjnym ostatniego folda
y_pred_probs = ostatni_model.predict(X_val_fold)
ostatnie_y_pred = np.argmax(y_pred_probs, axis=1)
ostatnie_y_test_classes = np.argmax(ostatnie_y_test, axis=1)

# Rozszerzona ocena: Raport klasyfikacji dla ostatniego folda
print("\nRaport klasyfikacji dla ostatniego folda:")
print(classification_report(ostatnie_y_test_classes, ostatnie_y_pred, target_names=classes))

# Wykresy strat i dokładności dla ostatniego folda
plt.figure(figsize=(12, 5))

# Wykres strat
plt.subplot(1, 2, 1)
plt.plot(ostatni_history.history['loss'], label='Strata treningowa')
plt.plot(ostatni_history.history['val_loss'], label='Strata walidacyjna')
plt.title('Krzywa strat (ostatni fold)')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(ostatni_history.history['accuracy'], label='Dokładność treningowa')
plt.plot(ostatni_history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Krzywa dokładności (ostatni fold)')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()

plt.tight_layout()
plt.show()

# Macierz pomyłek dla ostatniego folda
cm = confusion_matrix(ostatnie_y_test_classes, ostatnie_y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    title='Macierz pomyłek (ostatni fold)',
    ylabel='Rzeczywista klasa',
    xlabel='Predykcja'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.show()

# Analiza rozkładu danych według klas: Histogramy odległości
plt.figure(figsize=(8, 6))
for klasa, nazwa in enumerate(classes):
    plt.hist(odleglosci[etykiety == klasa], bins=50, alpha=0.5, label=nazwa)
plt.xlabel('Odległość (m)')
plt.ylabel('Liczba próbek')
plt.title('Rozkład odległości według klas')
plt.legend()
plt.show()
