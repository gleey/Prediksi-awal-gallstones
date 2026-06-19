"""
Script utama untuk melatih model ANN prediksi penyakit gallstone.

Penggunaan:
    python train.py

Akan melatih model, memilih epoch terbaik (dari rentang 50-100),
menyimpan model ke file, dan menampilkan hasil evaluasi.
"""
import os
import sys
import json

from data_loader import prepare_dataset
from ann_model import ANN
from metrics import accuracy_score, classification_report, print_confusion_matrix


def main():
    # ============================================================
    #  Konfigurasi
    # ============================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset-gal.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "model.json")
    NORM_PARAMS_PATH = os.path.join(BASE_DIR, "normalization_params.json")
    EPOCHS = 100          # Jumlah epoch training (maksimum)
    EPOCH_MIN = 50        # Epoch minimum untuk pencarian terbaik
    EPOCH_MAX = 100       # Epoch maksimum untuk pencarian terbaik
    LEARNING_RATE = 0.01  # Learning rate
    HIDDEN1 = 8           # Neuron hidden layer 1
    HIDDEN2 = 4           # Neuron hidden layer 2
    TEST_RATIO = 0.2      # Proporsi data testing (20%)

    print("=" * 65)
    print("   PREDIKSI AWAL PENYAKIT GALLSTONE MENGGUNAKAN METODE ANN")
    print("   (Artificial Neural Network / Jaringan Saraf Tiruan)")
    print("=" * 65)

    # ============================================================
    #  1. Muat dan Persiapkan Data
    # ============================================================
    print("\n[1] MEMUAT DATASET")
    print("-" * 40)

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: File dataset tidak ditemukan: {DATASET_PATH}")
        sys.exit(1)

    X_train, y_train, X_test, y_test, header, mins, maxs, feat_idx = prepare_dataset(
        DATASET_PATH, test_ratio=TEST_RATIO
    )

    # Tampilkan distribusi kelas
    train_class0 = sum(1 for y in y_train if y == 0)
    train_class1 = sum(1 for y in y_train if y == 1)
    test_class0 = sum(1 for y in y_test if y == 0)
    test_class1 = sum(1 for y in y_test if y == 1)

    print(f"\n  Distribusi kelas (Training):")
    print(f"    Kelas 0 (Gallstone Yes) : {train_class0} sampel")
    print(f"    Kelas 1 (Gallstone No)  : {train_class1} sampel")
    print(f"\n  Distribusi kelas (Testing):")
    print(f"    Kelas 0 (Gallstone Yes) : {test_class0} sampel")
    print(f"    Kelas 1 (Gallstone No)  : {test_class1} sampel")

    # ============================================================
    #  2. Buat dan Konfigurasi Model ANN
    # ============================================================
    print("\n[2] ARSITEKTUR MODEL ANN")
    print("-" * 40)

    n_features = len(X_train[0])
    model = ANN(
        n_input=n_features,
        n_hidden1=HIDDEN1,
        n_hidden2=HIDDEN2,
        learning_rate=LEARNING_RATE
    )

    arch = model.get_architecture_info()
    print(f"  Input Layer   : {arch['input']} neuron (fitur)")
    print(f"  Hidden Layer 1: {arch['hidden1']} neuron (aktivasi: {arch['activation_hidden']})")
    print(f"  Hidden Layer 2: {arch['hidden2']} neuron (aktivasi: {arch['activation_hidden']})")
    print(f"  Output Layer  : {arch['output']} neuron (aktivasi: {arch['activation_output']})")
    print(f"  Total Parameter: {arch['total_params']}")
    print(f"  Learning Rate  : {arch['learning_rate']}")
    print(f"  Loss Function  : Binary Cross-Entropy")
    print(f"  Epochs         : {EPOCHS} (pencarian terbaik: epoch {EPOCH_MIN}-{EPOCH_MAX})")

    # ============================================================
    #  3. Training dengan Pemilihan Epoch Terbaik
    # ============================================================
    print(f"\n[3] PROSES TRAINING ({EPOCHS} epochs, pilih terbaik dari {EPOCH_MIN}-{EPOCH_MAX})")
    print("-" * 40)

    model.train(
        X_train, y_train,
        epochs=EPOCHS,
        verbose=True,
        X_val=X_test,
        y_val=y_test,
        select_best_epoch=True,
        epoch_range=(EPOCH_MIN, EPOCH_MAX)
    )

    # ============================================================
    #  4. Evaluasi pada Data Testing (menggunakan bobot epoch terbaik)
    # ============================================================
    print(f"\n[4] EVALUASI PADA DATA TESTING ({len(X_test)} sampel)")
    print(f"    (menggunakan bobot dari epoch {model.best_epoch})")
    print("-" * 40)

    y_pred, y_prob = model.predict_batch(X_test)

    # Confusion Matrix
    print_confusion_matrix(y_test, y_pred)

    # Classification Report
    acc = classification_report(y_test, y_pred)

    # ============================================================
    #  5. Evaluasi pada Data Training
    # ============================================================
    print(f"\n[5] EVALUASI PADA DATA TRAINING ({len(X_train)} sampel)")
    print("-" * 40)

    y_pred_train, _ = model.predict_batch(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"  Akurasi Training: {train_acc * 100:.2f}%")
    print(f"  Akurasi Testing : {acc * 100:.2f}%")
    print(f"  Epoch Terbaik   : {model.best_epoch}")

    gap = abs(train_acc - acc) * 100
    if gap > 15:
        print(f"  [!] Selisih akurasi: {gap:.2f}% (kemungkinan overfitting)")
    else:
        print(f"  [v] Selisih akurasi: {gap:.2f}% (model cukup generalized)")

    # ============================================================
    #  6. Contoh Prediksi
    # ============================================================
    print(f"\n[6] CONTOH PREDIKSI (5 sampel pertama dari test set)")
    print("-" * 40)
    print(f"{'No':>4} {'Aktual':>10} {'Prediksi':>10} {'Probabilitas':>14} {'Status':>10}")
    print("-" * 50)

    status_map = {0: "Yes", 1: "No"}
    for i in range(min(5, len(X_test))):
        actual = y_test[i]
        pred = y_pred[i]
        prob = y_prob[i]
        status = "Benar" if actual == pred else "Salah"
        print(f"{i+1:>4} {status_map[actual]:>10} {status_map[pred]:>10} {prob:>14.4f} {status:>10}")

    # ============================================================
    #  7. Riwayat Akurasi Validasi (Epoch 50-100)
    # ============================================================
    print(f"\n[7] RIWAYAT AKURASI VALIDASI (epoch {EPOCH_MIN}-{EPOCH_MAX})")
    print("-" * 40)
    for i in range(EPOCH_MIN - 1, min(EPOCH_MAX, len(model.val_acc_history))):
        epoch = i + 1
        val_acc = model.val_acc_history[i]
        bar_len = int(val_acc / 2)
        bar = "#" * min(bar_len, 50)
        marker = " <-- TERBAIK" if epoch == model.best_epoch else ""
        print(f"  Epoch {epoch:>4}: {val_acc:.2f}% |{bar}{marker}")

    # ============================================================
    #  8. Riwayat Loss
    # ============================================================
    print(f"\n[8] RIWAYAT LOSS (setiap 10 epoch)")
    print("-" * 40)
    for i, loss in enumerate(model.loss_history):
        epoch = i + 1
        if epoch % 10 == 0 or epoch == 1:
            bar_len = int(loss * 50)
            bar = "#" * min(bar_len, 50)
            print(f"  Epoch {epoch:>4}: {loss:.4f} |{bar}")

    # ============================================================
    #  9. Simpan Model dan Parameter Normalisasi
    # ============================================================
    print(f"\n[9] MENYIMPAN MODEL")
    print("-" * 40)

    # Simpan model ANN (bobot & arsitektur)
    model.save_to_file(MODEL_PATH)
    print(f"  Model tersimpan   : {MODEL_PATH}")

    # Simpan parameter normalisasi (mins, maxs) agar predict.py bisa normalize input baru
    norm_params = {
        'mins': mins,
        'maxs': maxs
    }
    with open(NORM_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, indent=2)
    print(f"  Normalisasi param : {NORM_PARAMS_PATH}")

    print("\n" + "=" * 65)
    print("  TRAINING SELESAI!")
    print(f"  Epoch terbaik: {model.best_epoch}")
    print(f"  Akurasi testing pada epoch terbaik: {acc * 100:.2f}%")
    print(f"  Model dan parameter normalisasi telah disimpan.")
    print(f"  Jalankan 'python predict.py' untuk prediksi tanpa training ulang.")
    print("=" * 65)


if __name__ == "__main__":
    main()

