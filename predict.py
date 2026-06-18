"""
Script untuk melakukan prediksi penyakit gallstone secara interaktif.

Penggunaan:
    python predict.py

User akan diminta memasukkan 4 fitur penting pasien:
  1. Vitamin D
  2. C-Reactive Protein (CRP)
  3. Total Body Water (TBW)
  4. Lean Mass (LM) (%)
"""
import os
import sys

from data_loader import prepare_dataset, SELECTED_FEATURES
from ann_model import ANN


def create_trained_model(dataset_path):
    """
    Melatih model dan mengembalikan model beserta parameter normalisasi.
    """
    print("Melatih model terlebih dahulu...")
    X_train, y_train, X_test, y_test, header, mins, maxs, feat_idx = prepare_dataset(
        dataset_path, test_ratio=0.2
    )

    n_features = len(X_train[0])
    model = ANN(
        n_input=n_features,
        n_hidden1=8,
        n_hidden2=4,
        learning_rate=0.01
    )

    model.train(
        X_train, y_train,
        epochs=100,
        verbose=False,
        X_val=X_test,
        y_val=y_test,
        select_best_epoch=True,
        epoch_range=(50, 100)
    )

    print(f"Model selesai ditraining (epoch terbaik: {model.best_epoch}).\n")
    return model, header, mins, maxs, feat_idx


def normalize_input(raw_values, mins, maxs):
    """Normalisasi input baru menggunakan parameter dari data training."""
    normalized = []
    for i in range(len(raw_values)):
        range_val = maxs[i] - mins[i]
        if range_val == 0:
            normalized.append(0.0)
        else:
            normalized.append((raw_values[i] - mins[i]) / range_val)
    return normalized


def input_float(prompt):
    """Meminta input angka dari user dengan validasi."""
    while True:
        try:
            raw = input(f"  {prompt}: ").strip()
            if raw.lower() in ('q', 'quit', 'exit'):
                return None
            return float(raw)
        except ValueError:
            print("    >> Masukkan angka yang valid!")


# Penjelasan dan satuan untuk setiap fitur
FEATURE_INFO = {
    "Vitamin D": {
        "satuan": "ng/mL",
        "keterangan": "Kadar vitamin D dalam darah",
        "contoh": "Contoh: 20-50 ng/mL (normal)"
    },
    "C-Reactive Protein (CRP)": {
        "satuan": "mg/L",
        "keterangan": "Penanda inflamasi dalam darah",
        "contoh": "Contoh: < 3 mg/L (normal), > 10 mg/L (tinggi)"
    },
    "Total Body Water (TBW)": {
        "satuan": "Liter",
        "keterangan": "Total cairan tubuh",
        "contoh": "Contoh: 30-60 L (tergantung berat badan)"
    },
    "Lean Mass (LM) (%)": {
        "satuan": "%",
        "keterangan": "Persentase massa tubuh tanpa lemak",
        "contoh": "Contoh: 50-90% (tergantung komposisi tubuh)"
    }
}


def get_patient_input():
    """Meminta input 4 fitur penting dari user."""
    values = []

    print("\n  Masukkan data pemeriksaan pasien:")
    print("  " + "-" * 50)

    for feat_name in SELECTED_FEATURES:
        info = FEATURE_INFO.get(feat_name, {})
        satuan = info.get("satuan", "")
        keterangan = info.get("keterangan", "")
        contoh = info.get("contoh", "")

        print(f"\n  [{len(values)+1}/4] {feat_name}")
        print(f"        {keterangan}")
        print(f"        {contoh}")

        val = input_float(f"{feat_name} ({satuan})")
        if val is None:
            return None
        values.append(val)

    return values


def main():
    DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset-gal.csv")

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: File dataset tidak ditemukan: {DATASET_PATH}")
        sys.exit(1)

    # Latih model
    model, header, mins, maxs, feat_idx = create_trained_model(DATASET_PATH)

    print("=" * 65)
    print("   PREDIKSI PENYAKIT GALLSTONE - INPUT DATA PASIEN")
    print("=" * 65)
    print("  Sistem menggunakan 4 fitur penting:")
    for i, feat in enumerate(SELECTED_FEATURES):
        print(f"    {i+1}. {feat}")
    print("\n  Ketik 'q' untuk keluar kapan saja.\n")

    pasien_no = 1

    while True:
        print(f"\n{'=' * 65}")
        print(f"  PASIEN #{pasien_no}")
        print(f"{'=' * 65}")

        # Ambil input dari user
        try:
            raw_values = get_patient_input()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Program dihentikan oleh user.")
            break

        if raw_values is None:
            print("\n  Program dihentikan oleh user.")
            break

        # Normalisasi input
        normalized = normalize_input(raw_values, mins, maxs)

        # Prediksi
        pred_class, pred_prob = model.predict(normalized)

        # Tampilkan hasil
        print(f"\n  {'=' * 50}")
        print(f"  HASIL PREDIKSI PASIEN #{pasien_no}")
        print(f"  {'=' * 50}")

        # Tampilkan data input
        print(f"  Data Input:")
        for i, feat in enumerate(SELECTED_FEATURES):
            satuan = FEATURE_INFO[feat]["satuan"]
            print(f"    {feat}: {raw_values[i]} {satuan}")

        print()
        if pred_class == 0:
            print(f"  Status       : POSITIF - Berisiko Gallstone")
        else:
            print(f"  Status       : NEGATIF - Tidak Berisiko Gallstone")

        print(f"  Probabilitas : {pred_prob:.4f}")
        print(f"  Confidence   : {abs(pred_prob - 0.5) * 200:.1f}%")

        if pred_class == 0:
            print(f"\n  -> Pasien BERISIKO terkena penyakit gallstone.")
            print(f"     Disarankan pemeriksaan lanjutan oleh dokter.")
        else:
            print(f"\n  -> Pasien TIDAK berisiko tinggi terkena gallstone.")

        print(f"\n  Catatan: Ini prediksi awal berbasis model ANN.")
        print(f"  Diagnosis akhir harus dilakukan oleh tenaga medis.")

        pasien_no += 1

        # Tanya apakah ingin prediksi lagi
        print(f"\n  {'-' * 50}")
        lanjut = input("  Prediksi pasien lain? (y/n): ").strip().lower()
        if lanjut not in ('y', 'ya', 'yes', ''):
            break

    print(f"\n{'=' * 65}")
    print(f"  Total pasien yang diprediksi: {pasien_no - 1}")
    print(f"  Terima kasih telah menggunakan sistem prediksi ANN!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
