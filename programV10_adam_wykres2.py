import os
import glob
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

        # -- Wczytywanie obrazu --
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

        # -- Wczytywanie Lidar --
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
# 3. Funkcje analityczne i wizualizacyjne
# -------------------------------------------------------------------------
def extract_features_for_random_forest(img_np, points_np):
    """
    Zamieniamy (image, points) na wektor cech 1D do użycia w RandomForest.
    Przykład:
      - spłaszczamy obraz (co będzie dość duże!) 
      - plus kilka statystyk z chmury punktów.
    W realnym zadaniu należałoby wymyślić sensowne cechy.
    """
    # 1) Obraz w kształcie [H, W, 3]; spłaszczamy
    # Uwaga: 224x224x3 = 150528 pikseli => to sporo, co spowalnia RandomForest
    flat_img = img_np.flatten()  # => shape (H*W*3,)

    # 2) Zróbmy np. statystyki z punktów Lidar:
    #    min/max/mean z x,y,z + średnia intensywność
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


def lidar_statistics(points):
    """
    Wyświetla statystyki zebranych punktów LiDAR.
    """
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]

    stats = {
        'x_mean': np.mean(x_vals),
        'x_median': np.median(x_vals),
        'x_std': np.std(x_vals),
        'y_mean': np.mean(y_vals),
        'y_median': np.median(y_vals),
        'y_std': np.std(y_vals),
        'z_mean': np.mean(z_vals),
        'z_median': np.median(z_vals),
        'z_std': np.std(z_vals),
    }

    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

def segment_points(points, distance_threshold):
    """
    Segmentacja punktów na podstawie odległości od czujnika.
    """
    distances = np.linalg.norm(points[:, :3], axis=1)
    close_points = points[distances < distance_threshold]
    far_points = points[distances >= distance_threshold]
    return close_points, far_points

def detect_anomalies(points, threshold=3.0):
    """
    Wykrywanie anomalii (punkty daleko od średniej odległości).
    """
    distances = np.linalg.norm(points[:, :3], axis=1)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    anomalies = points[np.abs(distances - mean_distance) > threshold * std_distance]
    return anomalies

def analyze_density(points, grid_size=1.0):
    """
    Analiza gęstości punktów w przestrzeni.
    """
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]

    x_bins = np.arange(np.min(x_vals), np.max(x_vals) + grid_size, grid_size)
    y_bins = np.arange(np.min(y_vals), np.max(y_vals) + grid_size, grid_size)
    z_bins = np.arange(np.min(z_vals), np.max(z_vals) + grid_size, grid_size)

    density, _ = np.histogramdd(points[:, :3], bins=(x_bins, y_bins, z_bins))
    return density

def plot_distances_with_anomalies(points, anomalies):
    """
    Wykres ukazujący średnią odległość pomiaru oraz zaznaczenie punktów anomalii.
    """
    distances = np.linalg.norm(points[:, :3], axis=1)
    anomaly_distances = np.linalg.norm(anomalies[:, :3], axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, alpha=0.5, label='All Points')
    plt.hist(anomaly_distances, bins=50, alpha=0.5, label='Anomalies')
    plt.axvline(np.mean(distances), color='r', linestyle='dashed', linewidth=1, label='Mean Distance')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Distances with Anomalies Highlighted')
    plt.show()

def calculate_differences(points):
    """
    Wyznacza różnice w wartościach między kolejnymi obserwacjami.
    """
    differences = np.diff(points, axis=0)
    return differences

def plot_differences(differences):
    """
    Wizualizuje różnice na osobnych wykresach liniowych.
    """
    labels = ['x', 'y', 'z', 'intensity']
    plt.figure(figsize=(12, 10))
    
    for i, label in enumerate(labels):
        plt.subplot(4, 1, i + 1)
        plt.plot(differences[:, i], label=f'Difference in {label}')
        plt.xlabel('Observation Index')
        plt.ylabel('Difference')
        plt.legend()
        plt.title(f'Differences Between Consecutive Observations in {label}')
    
    plt.tight_layout()
    plt.show()

def plot_seasonal_boxplot(points):
    """
    Wykres pudełkowy dla sezonów.
    """
    df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity'])
    df['season'] = (df.index % 4) + 1  # Przykładowa kolumna sezonu (1-4)
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column='intensity', by='season')
    plt.title('Box Plot of Intensity by Season')
    plt.suptitle('')
    plt.xlabel('Season')
    plt.ylabel('Intensity')
    plt.show()

def plot_polar_plot(points):
    """
    Wykres polarowy.
    """
    df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity'])
    df['angle'] = np.arctan2(df['y'], df['x'])
    df['radius'] = np.sqrt(df['x']**2 + df['y']**2)
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    sc = ax.scatter(df['angle'], df['radius'], c=df['intensity'], cmap='viridis', alpha=0.75)
    plt.colorbar(sc, label='Intensity')
    plt.title('Polar Plot of Points')
    plt.show()
    
    # Display table with point ID, angle, distance, and intensity
    df['id'] = df.index
    table_df = df[['id', 'angle', 'radius', 'intensity']]
    print(table_df)
    
    # Save table to CSV
    table_df.to_csv('polar_plot_data.csv', index=False)

def main():
    root_dir = r"C:\program_przejsciowa\dane_przejsciowa"  # Dostosuj do siebie
    ds = create_pandaset_tf_dataset(root_dir, resize=(224,224), shuffle_buf=512)

    # Przykładowe użycie funkcji analitycznych i wizualizacyjnych
    for img_tf, points_tf in ds.take(1):
        points_np = points_tf.numpy()
        
        print("Statystyki LiDAR:")
        lidar_statistics(points_np)
        
        close_points, far_points = segment_points(points_np, distance_threshold=10.0)
        print(f"Liczba bliskich punktów: {len(close_points)}")
        print(f"Liczba dalekich punktów: {len(far_points)}")
        
        anomalies = detect_anomalies(points_np)
        print(f"Liczba anomalii: {len(anomalies)}")
        
        density = analyze_density(points_np)
        print(f"Gęstość punktów: {density}")

        # Plot distances with anomalies
        plot_distances_with_anomalies(points_np, anomalies)

        # Calculate and plot differences
        differences = calculate_differences(points_np)
        plot_differences(differences)

        # Seasonal analysis
        plot_seasonal_boxplot(points_np)
        plot_polar_plot(points_np)

if __name__ == "__main__":
    main()
