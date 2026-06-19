"""
Script untuk melakukan prediksi penyakit gallstone secara interaktif.

Penggunaan:
    python predict.py

CATATAN: Jalankan 'python train.py' terlebih dahulu untuk melatih
dan menyimpan model. Script ini langsung memuat model yang sudah
ditraining tanpa perlu melatih ulang.

User akan diminta memasukkan 4 fitur penting pasien:
  1. Vitamin D
  2. C-Reactive Protein (CRP)
  3. Total Body Water (TBW)
  4. Lean Mass (LM) (%)
"""
import os
import sys
import json

from data_loader import SELECTED_FEATURES
from ann_model import ANN


def load_trained_model(model_path, norm_params_path):
    """
    Memuat model ANN dan parameter normalisasi dari file yang sudah disimpan.

    Args:
        model_path: Path ke file model.json
        norm_params_path: Path ke file normalization_params.json

    Returns:
        model: Instance ANN dengan bobot yang sudah dimuat
        mins: List nilai minimum tiap fitur (untuk normalisasi)
        maxs: List nilai maksimum tiap fitur (untuk normalisasi)
    """
    # Muat model ANN
    model = ANN.load_from_file(model_path)

    # Muat parameter normalisasi
    with open(norm_params_path, 'r', encoding='utf-8') as f:
        norm_params = json.load(f)

    mins = norm_params['mins']
    maxs = norm_params['maxs']

    return model, mins, maxs


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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model.json")
    NORM_PARAMS_PATH = os.path.join(BASE_DIR, "normalization_params.json")

    # Cek apakah file model sudah ada
    if not os.path.exists(MODEL_PATH) or not os.path.exists(NORM_PARAMS_PATH):
        print("=" * 65)
        print("  ERROR: Model belum ditraining!")
        print("=" * 65)
        if not os.path.exists(MODEL_PATH):
            print(f"  File tidak ditemukan: {MODEL_PATH}")
        if not os.path.exists(NORM_PARAMS_PATH):
            print(f"  File tidak ditemukan: {NORM_PARAMS_PATH}")
        print(f"\n  Silakan jalankan 'python train.py' terlebih dahulu")
        print(f"  untuk melatih dan menyimpan model.")
        print("=" * 65)
        sys.exit(1)

    # Muat model yang sudah ditraining (tanpa perlu training ulang)
    print("Memuat model yang sudah ditraining...")
    model, mins, maxs = load_trained_model(MODEL_PATH, NORM_PARAMS_PATH)
    print(f"Model berhasil dimuat (epoch terbaik: {model.best_epoch}).\n")

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
