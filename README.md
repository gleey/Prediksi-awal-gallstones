# Prediksi Awal Penyakit Gallstone Menggunakan Metode ANN

Implementasi **Artificial Neural Network (ANN)** / Jaringan Saraf Tiruan untuk prediksi awal penyakit gallstone (batu empedu) menggunakan Python murni tanpa library machine learning eksternal.

## Deskripsi

Proyek ini membangun model klasifikasi biner untuk memprediksi apakah seorang pasien berisiko terkena penyakit gallstone berdasarkan data klinis. Model ANN diimplementasikan dari nol (*from scratch*) menggunakan Python standar.

Berdasarkan penelitian, sistem ini menggunakan **4 fitur penting** yang paling berpengaruh dalam prediksi gallstone:

| No | Fitur | Satuan | Keterangan |
|----|-------|--------|------------|
| 1 | Vitamin D | ng/mL | Kadar vitamin D dalam darah |
| 2 | C-Reactive Protein (CRP) | mg/L | Penanda inflamasi dalam darah |
| 3 | Total Body Water (TBW) | Liter | Total cairan tubuh |
| 4 | Lean Mass (LM) | % | Persentase massa tubuh tanpa lemak |

## Arsitektur Model

```
Input Layer  : 4 neuron (fitur)
Hidden Layer 1 : 8 neuron (aktivasi: ReLU)
Hidden Layer 2 : 4 neuron (aktivasi: ReLU)
Output Layer : 1 neuron (aktivasi: Sigmoid)
```

- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Gradient Descent
- **Inisialisasi Bobot**: He Initialization
- **Normalisasi**: Min-Max Scaling
- **Pemilihan Epoch**: Epoch terbaik dipilih dari rentang 50-100 berdasarkan akurasi validasi

## Struktur File

```
Prediksi-awal-gallstones/
├── dataset-gal.csv   # Dataset gallstone (319 sampel, 39 kolom)
├── data_loader.py    # Memuat CSV, seleksi fitur, normalisasi, split data
├── ann_model.py      # Implementasi ANN (forward, backward, training)
├── metrics.py        # Metrik evaluasi (confusion matrix, precision, recall, F1)
├── train.py          # Script utama untuk training dan evaluasi model
├── predict.py        # Script prediksi interaktif untuk data pasien baru
└── README.md
```

## Cara Menjalankan

### Prasyarat

- Python 3.x (tanpa library tambahan)

### Training Model

```bash
python train.py
```

Akan menampilkan:
- Informasi dataset dan distribusi kelas
- Arsitektur model ANN
- Progress training per epoch (loss dan akurasi)
- Pemilihan epoch terbaik (dari epoch 50-100)
- Confusion matrix dan laporan klasifikasi
- Riwayat akurasi validasi dan loss

### Prediksi Interaktif

```bash
python predict.py
```

User akan diminta memasukkan 4 nilai pemeriksaan pasien:
1. Vitamin D (ng/mL)
2. C-Reactive Protein / CRP (mg/L)
3. Total Body Water / TBW (Liter)
4. Lean Mass / LM (%)

Sistem akan memberikan hasil prediksi berupa status risiko gallstone beserta tingkat confidence.

## Class Labels

| Fitur | Nilai | Keterangan |
|-------|-------|------------|
| Gallstone Status | 0 | Yes (Positif) |
| Gallstone Status | 1 | No (Negatif) |
| Gender | 0 / 1 | Male / Female |
| Comorbidity | 0-3 | Jumlah kondisi komorbid |
| Coronary Artery Disease | 0 / 1 | No / Yes |
| Hypothyroidism | 0 / 1 | No / Yes |
| Hyperlipidemia | 0 / 1 | No / Yes |
| Diabetes Mellitus | 0 / 1 | No / Yes |
| Hepatic Fat Accumulation | 0-4 | Grade 0 (none) - Grade 4 (very severe) |

## Hasil Evaluasi

| Metrik | Nilai |
|--------|-------|
| Akurasi Testing | ~73% |
| Akurasi Training | ~74% |
| Total Parameter | 81 |
| Epoch Terbaik | ~78 |

## Sumber Dataset

Esen, I., Arslan, H., Aktürk, S., Gülşen, M., Kültekin, N., & Özdemir, O. (2024). Gallstone [Dataset]. [https://archive.ics.uci.edu/dataset/1150/gallstone-1] (https://archive.ics.uci.edu/dataset/1150/gallstone-1)

UCI Machine Learning Repository. [https://doi.org/10.1097/md.0000000000037258](https://doi.org/10.1097/md.0000000000037258)
