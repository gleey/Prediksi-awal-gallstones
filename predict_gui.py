"""
Aplikasi GUI Desktop (Tkinter) untuk Prediksi Awal Gallstone.
Menyediakan antarmuka grafis native untuk mengisi 38 fitur pasien,
memuat sampel acak dari dataset, dan melihat hasil prediksi ANN.

Penggunaan:
    python predict_gui.py
"""
import os
import sys
import json
import random
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess
import threading

from ann_model import ANN
from data_loader import SELECTED_FEATURES, load_csv

# Path konfigurasi model dan dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.json")
NORM_PARAMS_PATH = os.path.join(BASE_DIR, "normalization_params.json")
DATASET_PATH = os.path.join(BASE_DIR, "dataset-gal.csv")

# Kategori Fitur untuk Layout 3 Kolom
CAT_DEMO = [
    "Age", "Gender", "Comorbidity", "Coronary Artery Disease (CAD)", "Hypothyroidism",
    "Hyperlipidemia", "Diabetes Mellitus (DM)", "Height", "Weight", "Body Mass Index (BMI)",
    "Obesity (%)"
]

CAT_BODY = [
    "Total Body Water (TBW)", "Extracellular Water (ECW)", "Intracellular Water (ICW)",
    "Extracellular Fluid/Total Body Water (ECF/TBW)", "Total Body Fat Ratio (TBFR) (%)",
    "Lean Mass (LM) (%)", "Body Protein Content (Protein) (%)", "Visceral Fat Rating (VFR)",
    "Bone Mass (BM)", "Muscle Mass (MM)", "Total Fat Content (TFC)", "Visceral Fat Area (VFA)",
    "Visceral Muscle Area (VMA) (Kg)", "Hepatic Fat Accumulation (HFA)"
]

CAT_LAB = [
    "Glucose", "Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)", "High Density Lipoprotein (HDL)",
    "Triglyceride", "Aspartat Aminotransferaz (AST)", "Alanin Aminotransferaz (ALT)",
    "Alkaline Phosphatase (ALP)", "Creatinine", "Glomerular Filtration Rate (GFR)",
    "C-Reactive Protein (CRP)", "Hemoglobin (HGB)", "Vitamin D"
]

# Teks Label & Satuan Bahasa Indonesia yang Rapi
LABEL_TEXTS = {
    "Age": "Umur (Tahun)",
    "Gender": "Jenis Kelamin",
    "Comorbidity": "Komorbiditas (Jumlah Penyakit)",
    "Coronary Artery Disease (CAD)": "Penyakit Jantung Koroner (CAD)",
    "Hypothyroidism": "Hipotiroidisme",
    "Hyperlipidemia": "Hiperlipidemia",
    "Diabetes Mellitus (DM)": "Diabetes Mellitus",
    "Height": "Tinggi Badan (cm)",
    "Weight": "Berat Badan (kg)",
    "Body Mass Index (BMI)": "Indeks Massa Tubuh (BMI)",
    "Obesity (%)": "Tingkat Obesitas (%)",
    
    "Total Body Water (TBW)": "Total Cairan Tubuh / TBW (L)",
    "Extracellular Water (ECW)": "Cairan Ekstrasel / ECW (L)",
    "Intracellular Water (ICW)": "Cairan Intrasel / ICW (L)",
    "Extracellular Fluid/Total Body Water (ECF/TBW)": "Rasio ECF/TBW (%)",
    "Total Body Fat Ratio (TBFR) (%)": "Rasio Lemak Tubuh (%)",
    "Lean Mass (LM) (%)": "Massa Tubuh Tanpa Lemak (%)",
    "Body Protein Content (Protein) (%)": "Protein Tubuh (%)",
    "Visceral Fat Rating (VFR)": "Rating Lemak Viseral (1-59)",
    "Bone Mass (BM)": "Massa Tulang (kg)",
    "Muscle Mass (MM)": "Massa Otot (kg)",
    "Total Fat Content (TFC)": "Total Kandungan Lemak (kg)",
    "Visceral Fat Area (VFA)": "Area Lemak Viseral",
    "Visceral Muscle Area (VMA) (Kg)": "Area Otot Viseral (kg)",
    "Hepatic Fat Accumulation (HFA)": "Akumulasi Lemak Hati",
    
    "Glucose": "Gula Darah / Glukosa (mg/dL)",
    "Total Cholesterol (TC)": "Kolesterol Total (mg/dL)",
    "Low Density Lipoprotein (LDL)": "Kolesterol LDL (mg/dL)",
    "High Density Lipoprotein (HDL)": "Kolesterol HDL (mg/dL)",
    "Triglyceride": "Trigliserida (mg/dL)",
    "Aspartat Aminotransferaz (AST)": "Enzim Hati AST (U/L)",
    "Alanin Aminotransferaz (ALT)": "Enzim Hati ALT (U/L)",
    "Alkaline Phosphatase (ALP)": "Enzim Hati ALP (U/L)",
    "Creatinine": "Kreatinin Darah (mg/dL)",
    "Glomerular Filtration Rate (GFR)": "GFR (Laju Filtrasi Ginjal)",
    "C-Reactive Protein (CRP)": "CRP / Penanda Inflamasi (mg/L)",
    "Hemoglobin (HGB)": "Hemoglobin Darah (g/dL)",
    "Vitamin D": "Kadar Vitamin D (ng/mL)"
}

# Opsi untuk Kolom Kategori (Combobox)
COMBO_OPTIONS = {
    "Gender": ["0 (Pria)", "1 (Wanita)"],
    "Comorbidity": ["0", "1", "2", "3"],
    "Coronary Artery Disease (CAD)": ["0 (Tidak)", "1 (Ya)"],
    "Hypothyroidism": ["0 (Tidak)", "1 (Ya)"],
    "Hyperlipidemia": ["0 (Tidak)", "1 (Ya)"],
    "Diabetes Mellitus (DM)": ["0 (Tidak)", "1 (Ya)"],
    "Hepatic Fat Accumulation (HFA)": ["0 (Tidak Ada)", "1 (Ringan)", "2 (Sedang)", "3 (Berat)", "4 (Sangat Berat)"]
}


class TrainingLogDialog(tk.Toplevel):
    """Jendela pop-up untuk log proses training ANN."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Proses Training Model ANN")
        self.geometry("600x400")
        self.transient(parent)
        self.grab_set()
        
        self.configure(bg="#f8fafc")
        
        lbl = ttk.Label(self, text="Log Proses Latihan (Training) Jaringan Saraf Tiruan:", font=("Segoe UI", 11, "bold"))
        lbl.pack(padx=15, pady=(15, 5), anchor="w")
        
        # Text Log Area
        self.text_frame = ttk.Frame(self)
        self.text_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        self.log_text = tk.Text(self.text_frame, bg="#0f172a", fg="#10b981", insertbackground="white",
                                font=("Consolas", 10), wrap="word")
        self.scrollbar = ttk.Scrollbar(self.text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.close_btn = ttk.Button(self, text="Tutup", state="disabled", command=self.destroy)
        self.close_btn.pack(pady=15)
        
        # Jalankan training
        self.start_training()

    def start_training(self):
        self.log_text.insert(tk.END, "Memulai training model dengan 38 fitur input...\n")
        self.log_text.insert(tk.END, "Mengeksekusi train.py...\n\n")
        self.log_text.see(tk.END)
        
        def run():
            try:
                # Jalankan train.py menggunakan interpreter python saat ini
                process = subprocess.Popen(
                    [sys.executable, "train.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=BASE_DIR
                )
                
                # Baca output secara real-time
                for line in process.stdout:
                    self.log_text.insert(tk.END, line)
                    self.log_text.see(tk.END)
                
                process.wait()
                
                if process.returncode == 0:
                    self.log_text.insert(tk.END, "\n" + "="*50 + "\n")
                    self.log_text.insert(tk.END, " TRAINING BERHASIL DISELESAIKAN!\n")
                    self.log_text.insert(tk.END, " Model baru telah disimpan ke model.json\n")
                    self.log_text.insert(tk.END, "="*50 + "\n")
                else:
                    self.log_text.insert(tk.END, f"\n[ERROR] Training gagal dengan kode: {process.returncode}\n")
            except Exception as e:
                self.log_text.insert(tk.END, f"\n[ERROR] Gagal menjalankan script training: {e}\n")
            finally:
                self.close_btn.config(state="normal")
                self.master.load_model_data()
                
        threading.Thread(target=run, daemon=True).start()


class GallstoneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Pakar Prediksi Awal Gallstones (ANN)")
        self.root.geometry("1220x820")
        self.root.minsize(1080, 720)
        
        self.model = None
        self.mins = None
        self.maxs = None
        self.widgets = {}
        
        # Setup Palet Warna Modern (Medical Slate)
        self.COLOR_BG = "#f1f5f9"       # Slate 100
        self.COLOR_CARD = "#ffffff"     # White
        self.COLOR_HEADER = "#0f172a"   # Slate 900
        self.COLOR_PRIMARY = "#2563eb"  # Blue 600
        self.COLOR_SUCCESS = "#16a34a"  # Green 600
        self.COLOR_DANGER = "#dc2626"   # Red 600
        self.COLOR_MUTED = "#64748b"    # Slate 500
        
        self.root.configure(bg=self.COLOR_BG)
        self.setup_styles()
        self.build_ui()
        
        # Muat Model dan Normalisasi
        self.load_model_data()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Override background defaults
        self.style.configure('.', background=self.COLOR_BG, foreground="#0f172a")
        
        # Frame & LabelFrame
        self.style.configure('TFrame', background=self.COLOR_BG)
        self.style.configure('Card.TFrame', background=self.COLOR_CARD, relief="flat", borderwidth=1)
        self.style.configure('TLabelFrame', background=self.COLOR_CARD, relief="groove", borderwidth=1)
        self.style.configure('TLabelFrame.Label', font=("Segoe UI", 10, "bold"), background=self.COLOR_CARD, foreground=self.COLOR_PRIMARY)
        
        # Labels
        self.style.configure('TLabel', font=("Segoe UI", 10), background=self.COLOR_BG)
        self.style.configure('Card.TLabel', background=self.COLOR_CARD)
        self.style.configure('CardHeader.TLabel', font=("Segoe UI", 12, "bold"), background=self.COLOR_CARD, foreground=self.COLOR_HEADER)
        
        # Header Labels
        self.style.configure('HeaderTitle.TLabel', font=("Segoe UI", 16, "bold"), background=self.COLOR_HEADER, foreground="#ffffff")
        self.style.configure('HeaderSub.TLabel', font=("Segoe UI", 10), background=self.COLOR_HEADER, foreground="#94a3b8")
        
        # Buttons
        self.style.configure('TButton', font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure('Primary.TButton', font=("Segoe UI", 10, "bold"), background=self.COLOR_PRIMARY, foreground="#ffffff")
        self.style.map('Primary.TButton', background=[('active', '#1d4ed8'), ('disabled', '#93c5fd')])
        
        self.style.configure('Success.TButton', font=("Segoe UI", 10, "bold"), background=self.COLOR_SUCCESS, foreground="#ffffff")
        self.style.map('Success.TButton', background=[('active', '#15803d')])
        
        self.style.configure('Danger.TButton', font=("Segoe UI", 10, "bold"), background=self.COLOR_DANGER, foreground="#ffffff")
        self.style.map('Danger.TButton', background=[('active', '#b91c1c')])
        
        self.style.configure('Secondary.TButton', font=("Segoe UI", 10), background=self.COLOR_MUTED, foreground="#ffffff")
        self.style.map('Secondary.TButton', background=[('active', '#475569')])

    def build_ui(self):
        # 1. Header Banner
        header_frame = tk.Frame(self.root, bg=self.COLOR_HEADER, height=85)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)
        
        lbl_title = ttk.Label(header_frame, text="Sistem Pakar Prediksi Awal Gallstones (Batu Empedu)", style="HeaderTitle.TLabel")
        lbl_title.pack(anchor="w", padx=20, pady=(12, 2))
        
        lbl_sub = ttk.Label(header_frame, text="Implementasi Jaringan Saraf Tiruan (ANN) menggunakan 38 Parameter Klinis Lengkap", style="HeaderSub.TLabel")
        lbl_sub.pack(anchor="w", padx=20, pady=(0, 10))
        
        # Main Work Area
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=3) # Kiri: Form Input
        main_frame.columnconfigure(1, weight=1) # Kanan: Hasil & Aksi
        main_frame.rowconfigure(0, weight=1)
        
        # ------------------------------------------------------------
        # PANE KIRI: 38 Input Fields (Scrollable)
        # ------------------------------------------------------------
        left_card = ttk.Frame(main_frame, style="Card.TFrame")
        left_card.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        
        # Card Inner Frame
        inner_left = ttk.Frame(left_card, style="Card.TFrame", padding=10)
        inner_left.pack(fill="both", expand=True)
        
        lbl_input_title = ttk.Label(inner_left, text="PARAMETER KLINIS PASIEN", style="CardHeader.TLabel")
        lbl_input_title.pack(anchor="w", pady=(0, 5))
        
        # Scrollable Canvas Setup
        self.canvas = tk.Canvas(inner_left, bg=self.COLOR_CARD, bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(inner_left, orient="vertical", command=self.canvas.yview)
        
        self.scroll_frame = ttk.Frame(self.canvas, style="Card.TFrame")
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to fill frame width
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bind MouseWheel scroll only when hovering left panel to prevent conflict
        self.canvas.bind('<Enter>', lambda _: self.canvas.bind_all('<MouseWheel>', self._on_mouse_wheel))
        self.canvas.bind('<Leave>', lambda _: self.canvas.unbind_all('<MouseWheel>'))
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Grid Kategori Form di dalam Scrollable Frame
        # Kita buat 3 LabelFrame Kategori Berdampingan secara Horisontal
        col1_frame = ttk.LabelFrame(self.scroll_frame, text=" 1. Demografi & Pemeriksaan Umum ")
        col1_frame.grid(row=0, column=0, padx=8, pady=5, sticky="nsew")
        
        col2_frame = ttk.LabelFrame(self.scroll_frame, text=" 2. Komposisi Cairan & Lemak ")
        col2_frame.grid(row=0, column=1, padx=8, pady=5, sticky="nsew")
        
        col3_frame = ttk.LabelFrame(self.scroll_frame, text=" 3. Uji Laboratorium & Darah ")
        col3_frame.grid(row=0, column=2, padx=8, pady=5, sticky="nsew")
        
        self.scroll_frame.columnconfigure(0, weight=1)
        self.scroll_frame.columnconfigure(1, weight=1)
        self.scroll_frame.columnconfigure(2, weight=1)
        
        # Populate Inputs
        self.populate_category_inputs(col1_frame, CAT_DEMO)
        self.populate_category_inputs(col2_frame, CAT_BODY)
        self.populate_category_inputs(col3_frame, CAT_LAB)
        
        # ------------------------------------------------------------
        # PANE KANAN: Aksi & Dashboard Hasil
        # ------------------------------------------------------------
        right_pane = ttk.Frame(main_frame)
        right_pane.grid(row=0, column=1, sticky="nsew")
        right_pane.rowconfigure(0, weight=1) # Aksi & Hasil Card
        right_pane.columnconfigure(0, weight=1)
        
        right_card = ttk.Frame(right_pane, style="Card.TFrame", padding=15)
        right_card.grid(row=0, column=0, sticky="nsew")
        
        # Card Header Kanan
        lbl_control_title = ttk.Label(right_card, text="AKSI & KONTROL SISTEM", style="CardHeader.TLabel")
        lbl_control_title.pack(anchor="w", pady=(0, 15))
        
        # Tombol Aksi
        btn_frame = ttk.Frame(right_card, style="Card.TFrame")
        btn_frame.pack(fill="x", pady=(0, 20))
        
        self.sample_btn = ttk.Button(btn_frame, text="🎲  MUAT SAMPEL ACAK", style="Success.TButton", command=self.load_random_sample)
        self.sample_btn.pack(fill="x", pady=5)
        
        self.predict_btn = ttk.Button(btn_frame, text="🔬  JALANKAN PREDIKSI ANN", style="Primary.TButton", command=self.run_prediction)
        self.predict_btn.pack(fill="x", pady=5)
        
        self.reset_btn = ttk.Button(btn_frame, text="🔄  KOSONGKAN FORM", style="Secondary.TButton", command=self.reset_fields)
        self.reset_btn.pack(fill="x", pady=5)
        
        # Separator
        sep = ttk.Separator(right_card, orient="horizontal")
        sep.pack(fill="x", pady=10)
        
        # HASIL ANALISIS DASHBOARD
        lbl_result_title = ttk.Label(right_card, text="DASHBOARD HASIL ANALISIS", style="CardHeader.TLabel")
        lbl_result_title.pack(anchor="w", pady=(5, 15))
        
        # Display Box Utama
        self.result_box = tk.Label(right_card, text="MENUNGGU INPUT", font=("Segoe UI", 13, "bold"),
                                   bg="#e2e8f0", fg=self.COLOR_MUTED, relief="flat", height=3, borderwidth=0)
        self.result_box.pack(fill="x", pady=(0, 15))
        
        # Metrik Detil
        metrics_frame = ttk.Frame(right_card, style="Card.TFrame")
        metrics_frame.pack(fill="x", pady=5)
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
        
        # Baris 1: Probabilitas Risiko
        lbl_prob_title = ttk.Label(metrics_frame, text="Probabilitas Risiko:", style="Card.TLabel", font=("Segoe UI", 10))
        lbl_prob_title.grid(row=0, column=0, sticky="w", pady=4)
        self.prob_lbl = ttk.Label(metrics_frame, text="-", style="Card.TLabel", font=("Segoe UI", 10, "bold"), foreground=self.COLOR_HEADER)
        self.prob_lbl.grid(row=0, column=1, sticky="e", pady=4)
        
        # Progress Bar Visualisasi Risiko
        self.prob_bar = ttk.Progressbar(right_card, orient="horizontal", mode="determinate")
        self.prob_bar.pack(fill="x", pady=(0, 10))
        
        # Baris 2: Confidence Model
        lbl_conf_title = ttk.Label(metrics_frame, text="Tingkat Kepercayaan (Confidence):", style="Card.TLabel", font=("Segoe UI", 10))
        lbl_conf_title.grid(row=1, column=0, sticky="w", pady=4)
        self.conf_lbl = ttk.Label(metrics_frame, text="-", style="Card.TLabel", font=("Segoe UI", 10, "bold"), foreground=self.COLOR_HEADER)
        self.conf_lbl.grid(row=1, column=1, sticky="e", pady=4)
        
        # Baris 3: Status Aktual (khusus sampel acak)
        self.lbl_actual_title = ttk.Label(metrics_frame, text="Status Aktual Pasien (Dataset):", style="Card.TLabel", font=("Segoe UI", 10))
        self.lbl_actual_title.grid(row=2, column=0, sticky="w", pady=4)
        self.actual_status_val = tk.StringVar(value="-")
        self.actual_status_lbl = ttk.Label(metrics_frame, textvariable=self.actual_status_val, style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        self.actual_status_lbl.grid(row=2, column=1, sticky="e", pady=4)
        
        # Baris 4: Kesesuaian Prediksi
        self.lbl_match_title = ttk.Label(metrics_frame, text="Kesesuaian Prediksi Model:", style="Card.TLabel", font=("Segoe UI", 10))
        self.lbl_match_title.grid(row=3, column=0, sticky="w", pady=4)
        self.match_val = tk.StringVar(value="-")
        self.match_lbl = ttk.Label(metrics_frame, textvariable=self.match_val, style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        self.match_lbl.grid(row=3, column=1, sticky="e", pady=4)
        
        # Box Rekomendasi Klinis
        lbl_recom_title = ttk.Label(right_card, text="Panduan Klinis Sementara:", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        lbl_recom_title.pack(anchor="w", pady=(10, 5))
        
        self.recom_text = tk.Text(right_card, bg="#f8fafc", fg="#334155", font=("Segoe UI", 9),
                                  wrap="word", height=7, borderwidth=1, relief="solid")
        self.recom_text.insert(tk.END, "Silakan isi parameter pasien di form sebelah kiri secara manual atau dengan mengklik 'Muat Sampel Acak', lalu jalankan prediksi model ANN untuk melihat panduan klinis.")
        self.recom_text.config(state="disabled")
        self.recom_text.pack(fill="x", pady=5)
        
        # ------------------------------------------------------------
        # BAR STATUS / FOOTER
        # ------------------------------------------------------------
        footer_frame = tk.Frame(self.root, bg="#e2e8f0", height=30)
        footer_frame.pack(fill="x", side="bottom")
        footer_frame.pack_propagate(False)
        
        self.status_bar = tk.Label(footer_frame, text="Memuat konfigurasi sistem...", font=("Segoe UI", 9),
                                   bg="#e2e8f0", fg="#475569", anchor="w")
        self.status_bar.pack(side="left", fill="both", expand=True, padx=15)
        
        self.train_btn = tk.Button(footer_frame, text="🛠️ Latih Model Baru (Train)", font=("Segoe UI", 9, "bold"),
                                   bg="#475569", fg="#ffffff", activebackground="#334155", activeforeground="#ffffff",
                                   bd=0, padx=10, relief="flat", command=self.open_training_log)
        self.train_btn.pack(side="right", fill="y")

    def populate_category_inputs(self, parent_frame, feature_list):
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.columnconfigure(1, weight=1)
        
        # Loop semua fitur di kategori ini
        for i, feat in enumerate(feature_list):
            clean_feat = feat.strip()
            
            # Label
            display_name = LABEL_TEXTS.get(clean_feat, clean_feat)
            lbl = ttk.Label(parent_frame, text=display_name + ":", style="Card.TLabel", wraplength=160, justify="left")
            lbl.grid(row=i, column=0, padx=(8, 4), pady=6, sticky="w")
            
            # Input Widget (Combobox atau Entry)
            if clean_feat in COMBO_OPTIONS:
                widget = ttk.Combobox(parent_frame, values=COMBO_OPTIONS[clean_feat], state="readonly", width=12)
                widget.grid(row=i, column=1, padx=(4, 8), pady=6, sticky="ew")
            else:
                widget = ttk.Entry(parent_frame, width=12)
                widget.grid(row=i, column=1, padx=(4, 8), pady=6, sticky="ew")
                
            self.widgets[feat] = widget

    # Event handlers & helper
    def _on_canvas_configure(self, event):
        # Set the width of the scroll_frame inside canvas to match canvas width
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_model_data(self):
        """Memuat bobot ANN dari model.json dan normalisasi params."""
        if os.path.exists(MODEL_PATH) and os.path.exists(NORM_PARAMS_PATH):
            try:
                self.model = ANN.load_from_file(MODEL_PATH)
                with open(NORM_PARAMS_PATH, 'r', encoding='utf-8') as f:
                    norm_params = json.load(f)
                self.mins = norm_params['mins']
                self.maxs = norm_params['maxs']
                self.status_bar.config(text=f"Model ANN berhasil dimuat. (Epoch Terbaik: {self.model.best_epoch})")
                self.predict_btn.config(state="normal")
            except Exception as e:
                self.status_bar.config(text=f"ERROR: Gagal memuat file model/normalisasi: {e}")
                self.predict_btn.config(state="disabled")
        else:
            self.status_bar.config(text="Model belum dilatih! Silakan klik 'Latih Model Baru' di pojok kanan bawah terlebih dahulu.")
            self.predict_btn.config(state="disabled")

    def open_training_log(self):
        """Membuka jendela popup log training."""
        TrainingLogDialog(self.root)

    def reset_fields(self):
        """Mengosongkan form input dan panel hasil."""
        for feat, widget in self.widgets.items():
            if isinstance(widget, ttk.Combobox):
                widget.set("")
            else:
                widget.delete(0, tk.END)
                
        # Reset labels
        self.result_box.config(text="MENUNGGU INPUT", bg="#e2e8f0", fg=self.COLOR_MUTED)
        self.prob_lbl.config(text="-")
        self.prob_bar.config(value=0)
        self.conf_lbl.config(text="-")
        self.actual_status_val.set("-")
        self.actual_status_lbl.config(foreground=self.COLOR_MUTED)
        self.match_val.set("-")
        self.match_lbl.config(foreground=self.COLOR_MUTED)
        
        self.recom_text.config(state="normal")
        self.recom_text.delete(1.0, tk.END)
        self.recom_text.insert(tk.END, "Silakan isi parameter pasien di form sebelah kiri secara manual atau dengan mengklik 'Muat Sampel Acak', lalu jalankan prediksi model ANN untuk melihat panduan klinis.")
        self.recom_text.config(state="disabled")

    def load_random_sample(self):
        """Memuat data pasien acak dari dataset-gal.csv."""
        if not os.path.exists(DATASET_PATH):
            messagebox.showerror("Error", f"File dataset tidak ditemukan di: {DATASET_PATH}")
            return
            
        try:
            header, dataset = load_csv(DATASET_PATH)
            row = random.choice(dataset)
            
            cleaned_header = [col.strip() for col in header]
            actual_class = int(row[0])  # Gallstone Status adalah kolom 0 (target)
            
            for feat in SELECTED_FEATURES:
                feat_clean = feat.strip()
                if feat_clean in cleaned_header:
                    idx = cleaned_header.index(feat_clean)
                    val = row[idx]
                    widget = self.widgets[feat]
                    
                    # Cek tipe widget
                    if isinstance(widget, ttk.Combobox):
                        # Cek integer untuk mapping
                        val_int = int(float(val))
                        options = widget['values']
                        # Set ke opsi yang berawalan angka tersebut
                        matched = False
                        for opt in options:
                            if opt.startswith(str(val_int)):
                                widget.set(opt)
                                matched = True
                                break
                        if not matched:
                            widget.set(str(val_int))
                    else:
                        # Entry text standard
                        widget.delete(0, tk.END)
                        val_float = float(val)
                        if val_float.is_integer():
                            widget.insert(0, str(int(val_float)))
                        else:
                            # Trim trailing zero desimal agar rapi
                            widget.insert(0, f"{val_float:.4f}".rstrip('0').rstrip('.'))
                            
            # Update label target aktual di UI
            self.actual_status_val.set("POSITIF (Berisiko)" if actual_class == 0 else "NEGATIF (Aman)")
            self.actual_status_lbl.config(foreground=self.COLOR_DANGER if actual_class == 0 else self.COLOR_SUCCESS)
            
            # Jalankan prediksi otomatis setelah memuat sampel acak
            self.run_prediction(auto=True, actual_class=actual_class)
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat baris data acak: {e}")

    def run_prediction(self, auto=False, actual_class=None):
        """Membaca parameter klinis, melakukan normalisasi, dan memprediksi dengan ANN."""
        if not self.model:
            messagebox.showwarning("Warning", "Model ANN belum siap atau belum dimuat. Silakan lakukan training terlebih dahulu.")
            return
            
        try:
            raw_values = []
            
            # Baca dan validasi ke-38 input dari widget
            for feat in SELECTED_FEATURES:
                widget = self.widgets[feat]
                val_str = widget.get().strip()
                
                if not val_str:
                    name = LABEL_TEXTS.get(feat, feat)
                    raise ValueError(f"Input '{name}' kosong. Silakan lengkapi data.")
                
                # Ekstrak nilai jika combobox menyimpan string berformat angka (misal "1 (Ya)" -> 1)
                if "(" in val_str:
                    val_str = val_str.split("(")[0].strip()
                
                try:
                    val = float(val_str)
                    raw_values.append(val)
                except ValueError:
                    name = LABEL_TEXTS.get(feat, feat)
                    raise ValueError(f"Input '{name}' harus bernilai angka.")
            
            if len(raw_values) != len(SELECTED_FEATURES):
                raise ValueError("Jumlah input tidak cocok dengan dimensi model!")
                
            # Jalankan normalisasi Min-Max
            normalized = []
            for i in range(len(raw_values)):
                range_val = self.maxs[i] - self.mins[i]
                if range_val == 0:
                    normalized.append(0.0)
                else:
                    normalized.append((raw_values[i] - self.mins[i]) / range_val)
                    
            # Prediksi dengan model ANN
            pred_class, pred_prob = self.model.predict(normalized)
            
            # Hitung persentase metrik untuk UI
            prob_percent = pred_prob * 100
            conf_percent = abs(pred_prob - 0.5) * 200
            
            # Tampilkan Hasil Analisis Utama
            if pred_class == 0:
                self.result_box.config(text="RISIKO TINGGI (POSITIF)", bg=self.COLOR_DANGER, fg="#ffffff")
                recom_text = "PASIEN BERISIKO TINGGI terkena penyakit batu empedu (gallstones).\n\nSaran Tindakan:\n1. Sangat disarankan untuk berkonsultasi dengan Dokter Spesialis Penyakit Dalam atau Bedah Digestif.\n2. Lakukan pemeriksaan pencitraan medis (misal USG Abdomen) untuk memverifikasi ada/tidaknya pembentukan batu empedu.\n3. Batasi asupan makanan berlemak tinggi, kolesterol tinggi, dan gorengan.\n4. Rutin olahraga ringan dan pantau kadar kolesterol serta gula darah."
            else:
                self.result_box.config(text="RISIKO RENDAH (NEGATIF)", bg=self.COLOR_SUCCESS, fg="#ffffff")
                recom_text = "PASIEN BERISIKO RENDAH terkena penyakit batu empedu (gallstones).\n\nSaran Tindakan:\n1. Pertahankan gaya hidup sehat dan pola makan seimbang rendah lemak jenuh.\n2. Konsumsi cairan/air putih yang cukup setiap hari untuk mencegah pengkristalan empedu.\n3. Lakukan medical check-up secara berkala untuk memantau fungsi hati (AST/ALT) dan kolesterol.\n4. Tingkatkan aktivitas fisik dan pertahankan berat badan ideal."
                
            # Update label metrik desimal
            self.prob_lbl.config(text=f"{prob_percent:.2f}%")
            self.prob_bar.config(value=int(prob_percent))
            self.conf_lbl.config(text=f"{conf_percent:.1f}%")
            
            # Tangani info status aktual untuk perbandingan data sampel
            if auto and actual_class is not None:
                match = "BENAR (COCOK)" if pred_class == actual_class else "SALAH (TIDAK COCOK)"
                fg_match = self.COLOR_SUCCESS if pred_class == actual_class else self.COLOR_DANGER
                self.match_val.set(match)
                self.match_lbl.config(foreground=fg_match)
            else:
                # Jika input diisi manual oleh user, kosongkan kolom status aktual
                self.actual_status_val.set("Diisi Manual")
                self.actual_status_lbl.config(foreground=self.COLOR_MUTED)
                self.match_val.set("-")
                self.match_lbl.config(foreground=self.COLOR_MUTED)
                
            # Update teks panduan klinis
            self.recom_text.config(state="normal")
            self.recom_text.delete(1.0, tk.END)
            self.recom_text.insert(tk.END, recom_text)
            self.recom_text.config(state="disabled")
            
        except ValueError as ve:
            messagebox.showerror("Validasi Gagal", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat memproses prediksi: {e}")


def main():
    root = tk.Tk()
    app = GallstoneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
