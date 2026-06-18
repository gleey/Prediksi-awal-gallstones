"""
Module untuk memuat dan memproses dataset gallstone.
Hanya menggunakan 4 fitur penting berdasarkan penelitian:
  1. Vitamin D
  2. C-Reactive Protein (CRP)
  3. Total Body Water (TBW)
  4. Lean Mass (LM)
"""
import csv
import random

# Fitur penting yang digunakan (nama kolom di CSV)
SELECTED_FEATURES = [
    "Vitamin D",
    "C-Reactive Protein (CRP)",
    "Total Body Water (TBW)",
    "Lean Mass (LM) (%)"
]


def load_csv(filepath):
    """Membaca file CSV dan mengembalikan header serta data numerik."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            try:
                numeric_row = [float(val) for val in row]
                data.append(numeric_row)
            except ValueError:
                continue  # skip baris yang tidak bisa dikonversi
    return header, data


def get_selected_indices(header):
    """Mendapatkan indeks kolom untuk fitur yang dipilih."""
    indices = []
    for feat_name in SELECTED_FEATURES:
        for i, col_name in enumerate(header):
            if feat_name == col_name:
                indices.append(i)
                break
        else:
            raise ValueError(f"Fitur '{feat_name}' tidak ditemukan di header CSV!")
    return indices


def normalize_minmax(data, feature_indices):
    """
    Normalisasi Min-Max untuk fitur yang dipilih.
    Mengembalikan data ternormalisasi serta min & max tiap fitur.
    """
    mins = []
    maxs = []
    for idx in feature_indices:
        col_values = [row[idx] for row in data]
        col_min = min(col_values)
        col_max = max(col_values)
        mins.append(col_min)
        maxs.append(col_max)

    normalized = []
    for row in data:
        new_row = list(row)
        for i, idx in enumerate(feature_indices):
            range_val = maxs[i] - mins[i]
            if range_val == 0:
                new_row[idx] = 0.0
            else:
                new_row[idx] = (row[idx] - mins[i]) / range_val
        normalized.append(new_row)

    return normalized, mins, maxs


def split_data(data, test_ratio=0.2, seed=42):
    """Membagi data menjadi train dan test set."""
    random.seed(seed)
    shuffled = list(data)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def prepare_dataset(filepath, test_ratio=0.2):
    """
    Pipeline lengkap: load CSV -> seleksi fitur penting -> normalisasi -> split.
    Mengembalikan X_train, y_train, X_test, y_test, header, mins, maxs, feature_indices.
    """
    header, data = load_csv(filepath)

    # Kolom 0 = target (Gallstone Status)
    target_idx = 0

    # Hanya gunakan fitur yang dipilih
    feature_indices = get_selected_indices(header)

    # Normalisasi fitur (bukan target)
    data, mins, maxs = normalize_minmax(data, feature_indices)

    # Split data
    train_data, test_data = split_data(data, test_ratio)

    # Pisahkan fitur dan target
    X_train = [[row[i] for i in feature_indices] for row in train_data]
    y_train = [int(row[target_idx]) for row in train_data]

    X_test = [[row[i] for i in feature_indices] for row in test_data]
    y_test = [int(row[target_idx]) for row in test_data]

    selected_names = [header[i] for i in feature_indices]

    print(f"Dataset dimuat: {len(data)} sampel total")
    print(f"  Training: {len(X_train)} sampel")
    print(f"  Testing : {len(X_test)} sampel")
    print(f"  Fitur   : {len(feature_indices)} fitur (dari 38 total)")
    print(f"  Fitur terpilih:")
    for name in selected_names:
        print(f"    - {name}")
    print(f"  Kelas   : 0 (Gallstone Yes), 1 (Gallstone No)")

    return X_train, y_train, X_test, y_test, header, mins, maxs, feature_indices
