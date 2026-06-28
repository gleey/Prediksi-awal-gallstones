# Prediksi Awal Penyakit Gallstone Menggunakan Metode ANN

Implementasi **Artificial Neural Network (ANN)** / Jaringan Saraf Tiruan untuk prediksi awal penyakit gallstone (batu empedu) menggunakan Python murni tanpa library machine learning eksternal.

## Deskripsi

Proyek ini membangun model klasifikasi biner untuk memprediksi apakah seorang pasien berisiko terkena penyakit gallstone berdasarkan data klinis. Model ANN diimplementasikan dari nol (*from scratch*) menggunakan Python standar.

Sistem ini menggunakan **38 fitur/parameter klinis lengkap** dari dataset untuk mendapatkan hasil klasifikasi yang komprehensif.

### Kategori Fitur (38 Fitur)

1. **Demografi & Pemeriksaan Umum**:
   - Umur (Age), Jenis Kelamin (Gender), Komorbiditas (Comorbidity), Penyakit Jantung Koroner (CAD), Hipotiroidisme, Hiperlipidemia, Diabetes Mellitus (DM), Tinggi Badan, Berat Badan, BMI, Tingkat Obesitas.
2. **Komposisi Cairan & Lemak Tubuh**:
   - Total Body Water (TBW), Extracellular Water (ECW), Intracellular Water (ICW), Rasio ECF/TBW, Rasio Lemak Tubuh (TBFR), Lean Mass (LM), Kandungan Protein Tubuh, Visceral Fat Rating (VFR), Massa Tulang, Massa Otot, Total Fat Content, Visceral Fat Area (VFA), Visceral Muscle Area (VMA), Hepatic Fat Accumulation (HFA).
3. **Uji Laboratorium & Darah**:
   - Gula Darah (Glucose), Kolesterol Total (TC), LDL, HDL, Trigliserida, Enzim AST, Enzim ALT, Enzim ALP, Kreatinin, GFR, C-Reactive Protein (CRP), Hemoglobin (HGB), Vitamin D.

---

## Arsitektur Model

```
Input Layer     : 38 neuron (fitur)
Hidden Layer 1  : 16 neuron (aktivasi: ReLU)
Hidden Layer 2  : 8 neuron (aktivasi: ReLU)
Output Layer    : 1 neuron (aktivasi: Sigmoid)
```

- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Gradient Descent
- **Inisialisasi Bobot**: He Initialization
- **Normalisasi**: Min-Max Scaling
- **Pemilihan Epoch**: Epoch terbaik dipilih dari rentang 50-100 berdasarkan akurasi validasi tertinggi

---

## Struktur File

```
Prediksi-awal-gallstones/
├── dataset-gal.csv            # Dataset gallstone (319 sampel, 39 kolom)
├── data_loader.py             # Memuat CSV, normalisasi, dan split data train/test
├── ann_model.py               # Implementasi ANN (forward, backward, save/load)
├── metrics.py                 # Evaluasi model (confusion matrix, precision, recall, F1)
├── train.py                   # Script training model & pencarian epoch terbaik
├── predict_gui.py             # GUI Desktop (Tkinter) native dengan pemuatan sampel acak
└── README.md                  # Dokumentasi proyek
```

---

## Cara Menjalankan

### Prasyarat

- Python 3.x (tanpa memerlukan instalasi library ML eksternal seperti TensorFlow/Scikit-Learn)

### 1. Training Model

Jalankan perintah berikut untuk melatih model ANN baru pada data training:

```bash
python train.py
```

Akan menampilkan:
- Progress loss dan akurasi tiap epoch.
- Grafik visual sederhana nilai akurasi testing pada epoch 50-100.
- Confusion Matrix lengkap & Laporan Klasifikasi (Precision, Recall, F1-Score).
- Penyimpanan model ke `model.json` dan parameter normalisasi desimal ke `normalization_params.json`.

### 2. Menjalankan GUI Desktop (Tkinter)

Jalankan perintah berikut untuk membuka aplikasi desktop native (tanpa perlu web/localhost):

```bash
python predict_gui.py
```

**Fitur Unggulan GUI Desktop:**
- **🎲 Muat Sampel Acak**: Klik tombol ini untuk otomatis memuat satu baris data pasien secara acak dari dataset. Semua 38 field input akan terisi otomatis, prediksi ANN dijalankan, dan Anda dapat melihat apakah hasil prediksi tersebut **Cocok (Benar/Salah)** dengan status aktual pasien dari dataset.
- **🔬 Jalankan Prediksi**: Untuk menganalisis input data pasien baru yang dimasukkan secara manual.
- **🔄 Kosongkan Form**: Mereset seluruh input field dan panel hasil.
- **🛠️ Latih Model Baru**: Melatih ulang model langsung dari tombol di footer GUI, dilengkapi dengan jendela pemantauan log proses training.

### 3. Prediksi Interaktif Terminal

Jika Anda ingin melakukan prediksi secara interaktif satu per satu lewat baris perintah terminal:

```bash
python predict.py
```

---

## Hasil Evaluasi Model Terbaik

Berdasarkan pencarian epoch terbaik (Epoch 81):

| Metrik | Nilai |
|--------|-------|
| Akurasi Training | 80.78% |
| Akurasi Testing | 71.88% |
| Total Parameter | 769 |
| Epoch Terbaik | 81 |

---

## Sumber Dataset

Esen, I., Arslan, H., Aktürk, S., Gülşen, M., Kültekin, N., & Özdemir, O. (2024). Gallstone [Dataset]. [https://archive.ics.uci.edu/dataset/1150/gallstone-1](https://archive.ics.uci.edu/dataset/1150/gallstone-1)

UCI Machine Learning Repository. [https://doi.org/10.1097/md.0000000000037258](https://doi.org/10.1097/md.0000000000037258)
