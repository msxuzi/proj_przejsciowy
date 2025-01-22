import os
import glob
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

# Dodatkowa biblioteka do klasyfikacji (Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# 1. Klasa do odczytu plików Pandaset (bez trenowania sieci)
# -------------------------------------------------------------------------
class PandasetReader:
    """
    Zarządza strukturą plików:
      root_dir/
      ├─ 001/
      │   ├─ camera/front_camera/XX.jpg
      │   ├─ lidar/XX.pkl (z kolumnami [x, y, z, i, t, d])
      ├─ 002/
      ├─ ...
      └─ 047/

    Załóżmy, że każda scena ma 80 klatek (00..79).
    W metodzie get_item zwracamy (img_rgb, points) – 
    gdzie `img_rgb` to [H, W, 3], a `points` to [N, 4].
    """
    def __init__(self, root_dir, resize=(224, 224)):
        self.root_dir = root_dir
        self.resize = resize

        # Zbierz foldery 001..047
        self.scene_dirs = sorted(
            glob.glob(os.path.join(self.root_dir, '[0-9][0-9][0-9]'))
        )
        # Stwórz listę (scene_int, frame_idx)
        self.samples = []
        for scene_path in self.scene_dirs:
            scene_str = os.path.basename(scene_path)  # np. "001"
            scene_int = int(scene_str)
            for frame_idx in range(80):
                self.samples.append((scene_int, frame_idx))

    def __len__(self):
        return len(self.samples)

    def get_item(self, index):
        """
        Zwraca (img_rgb, points), gdzie:
          - img_rgb: np.array (H, W, 3) float32, RGB
          - points:  np.array (N, 4) float32 (kolumny x,y,z,i)
        """
        scene_int, frame_idx = self.samples[index]
        scene_str = f"{scene_int:03d}"
        frame_str = f"{frame_idx:02d}"

        # -- Wczytywanie obrazu (przednia kamera) --
        cam_folder = os.path.join(self.root_dir, scene_str, "camera", "front_camera")
        img_path = os.path.join(cam_folder, f"{frame_str}.jpg")
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Brak obrazu: {img_path}")

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Nie wczytano obrazu {img_path}")

        # Skalowanie do (224,224) (lub innego rozmiaru)
        if self.resize is not None:
            w, h = self.resize
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # BGR -> RGB, float32
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32")

        # -- Wczytywanie LiDAR --
        lidar_folder = os.path.join(self.root_dir, scene_str, "lidar")
        lidar_path = os.path.join(lidar_folder, f"{frame_str}.pkl")
        if not os.path.isfile(lidar_path):
            raise FileNotFoundError(f"Brak Lidar: {lidar_path}")

        df = pd.read_pickle(lidar_path)
        # kolumny: x, y, z, i, t, d
        xyz = df[["x", "y", "z"]].to_numpy(dtype="float32")      # => [N,3]
        intensity = df["i"].to_numpy(dtype="float32")           # => [N]
        points = np.column_stack([xyz, intensity])              # => [N,4]

        return img_rgb, points


# -------------------------------------------------------------------------
# 2. Generator + tf.data.Dataset
# -------------------------------------------------------------------------
def pandaset_generator(reader):
    """Pythonowy generator -> (img_rgb, points)."""
    for idx in range(len(reader)):
        yield reader.get_item(idx)  # (img_np, points_np)

def create_pandaset_tf_dataset(root_dir, resize=(224,224), shuffle_buf=512):
    """
    Tworzy tf.data.Dataset z krotek (image, points).
    :param shuffle_buf: używamy mniejszego bufora w shuffle
    """
    reader = PandasetReader(root_dir, resize=resize)

    # Definiujemy "signature" - wymiary:
    # image: [H, W, 3], points: [None, 4]
    output_signature = (
        tf.TensorSpec(shape=(resize[1], resize[0], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: pandaset_generator(reader),
        output_signature=output_signature
    )

    # Shuffle z buforem = 512 (mniejszym niż len(reader))
    ds = ds.shuffle(buffer_size=shuffle_buf)

    return ds


# -------------------------------------------------------------------------
# 3. Funkcja do ekstrakcji cech
# -------------------------------------------------------------------------
def extract_features_for_random_forest(img_np, points_np):
    """
    Zamieniamy (image, points) na wektor cech 1D do użycia np. w RandomForest
    lub innej klasyfikacji. 
    Przykład:
      - spłaszczamy obraz (co będzie dość duże!),
      - plus kilka statystyk z chmury punktów.
    W realnym zadaniu należałoby wymyślić bardziej adekwatne cechy.
    """
    # 1) Obraz w kształcie [H, W, 3]; spłaszczamy
    flat_img = img_np.flatten()  # => shape (H*W*3,)

    # 2) Zróbmy np. statystyki z punktów Lidar:
    x_vals = points_np[:,0]
    y_vals = points_np[:,1]
    z_vals = points_np[:,2]
    i_vals = points_np[:,3]
    stats = np.array([
        x_vals.min(), x_vals.max(), x_vals.mean(),
        y_vals.min(), y_vals.max(), y_vals.mean(),
        z_vals.min(), z_vals.max(), z_vals.mean(),
        i_vals.min(), i_vals.max(), i_vals.mean()
    ], dtype="float32")

    # 3) Łączymy w jeden wektor
    features = np.concatenate([flat_img, stats])
    return features


# -------------------------------------------------------------------------
# 4. Funkcja do obliczania bardziej wiarygodnej odległości
# -------------------------------------------------------------------------
def compute_filtered_distance(points_np, intensity_threshold=0.1, percentile=5):
    """
    Oblicza wiarygodną odległość, filtrując szumy:
    - Odrzuca punkty o intensywności poniżej intensity_threshold.
    - Zamiast najmniejszej odległości, używa percentyla najbliższych odległości.
    """
    # Filtruj punkty o niskiej intensywności
    high_quality_points = points_np[points_np[:, 3] >= intensity_threshold]
    
    # Jeśli nie ma punktów po filtracji, zwróć nieskończoność lub odpowiednią wartość
    if len(high_quality_points) == 0:
        return np.inf

    # Oblicz odległości dla przefiltrowanych punktów
    distances = np.linalg.norm(high_quality_points[:, :3], axis=1)
    
    # Użyj percentyla jako reprezentatywnej odległości zamiast minimum
    reliable_distance = np.percentile(distances, percentile)
    return reliable_distance


# -------------------------------------------------------------------------
# 5. Główna funkcja main() z przykładowym pipeline
# -------------------------------------------------------------------------
def main():
    root_dir = r"C:\program_przejsciowa\dane_przejsciowa"  # Dostosuj do swojej ścieżki
    ds = create_pandaset_tf_dataset(root_dir, resize=(224,224), shuffle_buf=512)

    # Ustalamy progi odległości do klasyfikacji
    danger_threshold = 15.0  # m - poniżej tego: klasa niebezpieczna
    risk_threshold = 30.0    # m - poniżej tego: klasa ryzyka, powyżej: bezpieczna

    # ======== TUTAJ MOŻNA ZMIENIĆ ILOŚĆ OBLICZANYCH PRÓBEK =========
    max_samples = 1000000
    # =================================================================

    X_list = []
    y_list = []
    distances_list = []  # Lista do przechowywania odfiltrowanych odległości

    # Zbieramy próbki w pętli
    i = 0
    for img_tf, points_tf in ds:
        # Konwersja z tensora TF do numpy
        img_np = img_tf.numpy()       
        points_np = points_tf.numpy() 

        # Wyciągamy cechy
        feats = extract_features_for_random_forest(img_np, points_np)

        # Obliczamy bardziej wiarygodną odległość z filtrowaniem szumów
        filtered_dist = compute_filtered_distance(points_np, intensity_threshold=0.1, percentile=5)

        # Zapamiętujemy odfiltrowaną odległość do zapisu do CSV
        distances_list.append(filtered_dist)

        # Przydzielamy etykietę (0, 1, 2) w zależności od odległości
        if filtered_dist < danger_threshold:
            label = 0  # niebezpieczna odległość
        elif filtered_dist < risk_threshold:
            label = 1  # klasa ryzyka
        else:
            label = 2  # bezpieczna odległość

        X_list.append(feats)
        y_list.append(label)

        i += 1
        if i >= max_samples:
            break

    # Konwertujemy do tablic NumPy
    X = np.vstack(X_list)
    y = np.array(y_list, dtype="int")

    print("Shape cech (X):", X.shape)
    print("Shape etykiet (y):", y.shape)

    # Zapisujemy odfiltrowane odległości do pliku CSV
    distances_csv_file = "filtered_distances.csv"
    df_distances = pd.DataFrame({"filtered_distance": distances_list})
    df_distances.to_csv(distances_csv_file, index=False)
    print(f"Odfiltrowane odległości zapisano do pliku: {distances_csv_file}")

    # ---------------------------------------------------------------------
    # (A) Przykład z klasyfikatorem Random Forest (opcjonalnie)
    # ---------------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    y_pred_rf = rf.predict(X)
    acc_rf = accuracy_score(y, y_pred_rf)
    print(f"Accuracy RandomForest (na zebranych danych): {acc_rf:.2f}")

    # ---------------------------------------------------------------------
    # (B) Sieć neuronowa - klasyfikacja 3 klas odległości
    # ---------------------------------------------------------------------
    # Podział danych na zbiory treningowe, walidacyjne i testowe
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("Shape treningowych (X_train):", X_train.shape)
    print("Shape walidacyjnych (X_val):", X_val.shape)
    print("Shape testowych (X_test):", X_test.shape)
    print("Shape etykiet treningowych (y_train):", y_train.shape)
    print("Shape etykiet walidacyjnych (y_val):", y_val.shape)
    print("Shape etykiet testowych (y_test):", y_test.shape)

    input_dim = X_train.shape[1]   # wymiar wejściowy (liczba cech)
    num_classes = 3                # trzy klasy odległości

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Trenujemy sieć neuronową z użyciem zbioru walidacyjnego
    history = model.fit(
        X_train, y_train,
        epochs=100, 
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Ewaluacja na zbiorze testowym
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Dokładność sieci neuronowej na zbiorze testowym: {accuracy:.2f}")

if __name__ == "__main__":
    main()
