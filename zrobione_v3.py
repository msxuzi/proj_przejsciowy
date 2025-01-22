import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Ścieżka do pliku CSV
sciezka = r"C:\program_przejsciowa\filtered_distances.csv"

# Wczytaj dane z pliku CSV
df = pd.read_csv(sciezka)

# Zakładamy, że plik CSV zawiera kolumnę 'odleglosc_w_metrach'
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

# Przygotowanie danych wejściowych dla sieci neuronowej
X = odleglosci.reshape(-1, 1)
y = to_categorical(etykiety, num_classes=3)

# Podział danych na zbiory treningowy, walidacyjny i testowy
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Budowa modelu sieci neuronowej
model = Sequential([
    Dense(32, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu i zapis historii treningu
history = model.fit(X_train, y_train,
                    epochs=150,
                    batch_size=128,
                    validation_data=(X_val, y_val))

# Ocena modelu na zbiorze testowym
wynik = model.evaluate(X_test, y_test)
print(f"Strata na zbiorze testowym: {wynik[0]:.4f}")
print(f"Dokładność na zbiorze testowym: {wynik[1]:.4f}")

# Predykcja na zbiorze testowym
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# 1. Rozszerzona ocena: Raport klasyfikacji
classes = ['Niebezpieczna', 'Ryzyko', 'Bezpieczna']
print("\nRaport klasyfikacji:")
print(classification_report(y_test_classes, y_pred, target_names=classes))

# 2. Wykresy strat i dokładności
plt.figure(figsize=(12, 5))

# Wykres strat
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Krzywa strat')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Krzywa dokładności')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()

plt.tight_layout()
plt.show()

# 3. Macierz pomyłek z użyciem Matplotlib
cm = confusion_matrix(y_test_classes, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    title='Macierz pomyłek',
    ylabel='Rzeczywista klasa',
    xlabel='Predykcja'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.show()

# 4. Analiza rozkładu danych według klas: Histogramy odległości
plt.figure(figsize=(8, 6))
for klasa, nazwa in enumerate(classes):
    plt.hist(odleglosci[etykiety == klasa], bins=50, alpha=0.5, label=nazwa)
plt.xlabel('Odległość (m)')
plt.ylabel('Liczba próbek')
plt.title('Rozkład odległości według klas')
plt.legend()
plt.show()
