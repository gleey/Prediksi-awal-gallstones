"""
Module untuk menghitung metrik evaluasi model.
"""


def confusion_matrix(y_true, y_pred, n_classes=2):
    """
    Menghitung confusion matrix.
    
    Returns:
        matrix: List 2D [n_classes x n_classes]
                matrix[actual][predicted]
    """
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def accuracy_score(y_true, y_pred):
    """Menghitung akurasi."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if len(y_true) > 0 else 0.0


def precision_score(y_true, y_pred, positive_class=0):
    """
    Menghitung precision untuk kelas positif tertentu.
    Precision = TP / (TP + FP)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, positive_class=0):
    """
    Menghitung recall untuk kelas positif tertentu.
    Recall = TP / (TP + FN)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, positive_class=0):
    """
    Menghitung F1-Score.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision_score(y_true, y_pred, positive_class)
    rec = recall_score(y_true, y_pred, positive_class)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def classification_report(y_true, y_pred, class_names=None):
    """
    Membuat laporan klasifikasi lengkap.
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        class_names: Nama kelas (opsional)
    """
    if class_names is None:
        class_names = {0: "Gallstone (Yes)", 1: "No Gallstone (No)"}

    classes = sorted(set(y_true) | set(y_pred))

    print("\n" + "=" * 65)
    print("                   LAPORAN KLASIFIKASI")
    print("=" * 65)
    print(f"{'Kelas':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 65)

    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for cls in classes:
        prec = precision_score(y_true, y_pred, cls)
        rec = recall_score(y_true, y_pred, cls)
        f1 = f1_score(y_true, y_pred, cls)
        support = sum(1 for t in y_true if t == cls)

        name = class_names.get(cls, f"Kelas {cls}")
        print(f"{name:<25} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")

        total_support += support
        weighted_precision += prec * support
        weighted_recall += rec * support
        weighted_f1 += f1 * support

    print("-" * 65)

    acc = accuracy_score(y_true, y_pred)
    print(f"{'Akurasi':<25} {'':>10} {'':>10} {acc:>10.4f} {total_support:>10}")
    if total_support > 0:
        print(f"{'Weighted Avg':<25} {weighted_precision/total_support:>10.4f} "
              f"{weighted_recall/total_support:>10.4f} {weighted_f1/total_support:>10.4f} "
              f"{total_support:>10}")

    print("=" * 65)

    return acc


def print_confusion_matrix(y_true, y_pred, class_names=None):
    """Menampilkan confusion matrix yang rapi."""
    if class_names is None:
        class_names = {0: "Yes(0)", 1: "No(1)"}

    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true) | set(y_pred))

    print("\n" + "=" * 45)
    print("            CONFUSION MATRIX")
    print("=" * 45)
    print(f"{'':>15}", end="")
    print("     Prediksi")
    print(f"{'':>15}", end="")
    for cls in classes:
        print(f"{class_names.get(cls, str(cls)):>10}", end="")
    print()
    print("-" * 45)

    for i, cls in enumerate(classes):
        label = f"Aktual {class_names.get(cls, str(cls))}"
        print(f"{label:>15}", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>10}", end="")
        print()

    print("=" * 45)

    # Penjelasan
    if len(classes) == 2:
        print(f"\n  TP (True Positive)  = {cm[0][0]:>4} (Gallstone terdeteksi benar)")
        print(f"  TN (True Negative)  = {cm[1][1]:>4} (Bukan gallstone terdeteksi benar)")
        print(f"  FP (False Positive) = {cm[1][0]:>4} (Salah prediksi sebagai gallstone)")
        print(f"  FN (False Negative) = {cm[0][1]:>4} (Gallstone tidak terdeteksi)")
