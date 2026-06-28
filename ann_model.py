"""
Implementasi Artificial Neural Network (ANN) dari nol menggunakan Python.
Arsitektur: Input → Hidden Layer 1 → Hidden Layer 2 → Output
Aktivasi : ReLU (hidden), Sigmoid (output)
Optimizer: Gradient Descent dengan learning rate
"""
import math
import random


# ============================================================
#  Fungsi Aktivasi
# ============================================================

def sigmoid(x):
    """Fungsi aktivasi sigmoid."""
    x = max(-500, min(500, x))  # clamp untuk menghindari overflow
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(output):
    """Turunan sigmoid berdasarkan output sigmoid."""
    return output * (1.0 - output)


def relu(x):
    """Fungsi aktivasi ReLU."""
    return max(0.0, x)


def relu_derivative(output):
    """Turunan ReLU."""
    return 1.0 if output > 0 else 0.0


# ============================================================
#  Kelas ANN
# ============================================================

class ANN:
    """
    Artificial Neural Network sederhana untuk klasifikasi biner.
    
    Arsitektur:
        Input Layer  → n_input neuron
        Hidden Layer 1 → n_hidden1 neuron (ReLU)
        Hidden Layer 2 → n_hidden2 neuron (ReLU)
        Output Layer → 1 neuron (Sigmoid)
    """

    def __init__(self, n_input, n_hidden1=16, n_hidden2=8, learning_rate=0.01, seed=42):
        """
        Inisialisasi jaringan.
        
        Args:
            n_input: Jumlah fitur input
            n_hidden1: Jumlah neuron hidden layer 1
            n_hidden2: Jumlah neuron hidden layer 2
            learning_rate: Learning rate untuk gradient descent
            seed: Random seed untuk reprodusibilitas
        """
        random.seed(seed)
        self.lr = learning_rate
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2

        # Inisialisasi bobot dengan He initialization (skala sqrt(2/n))
        # Layer 1: Input → Hidden1
        scale1 = math.sqrt(2.0 / n_input)
        self.weights_ih1 = [[random.gauss(0, scale1) for _ in range(n_input)] for _ in range(n_hidden1)]
        self.bias_h1 = [0.0] * n_hidden1

        # Layer 2: Hidden1 → Hidden2
        scale2 = math.sqrt(2.0 / n_hidden1)
        self.weights_h1h2 = [[random.gauss(0, scale2) for _ in range(n_hidden1)] for _ in range(n_hidden2)]
        self.bias_h2 = [0.0] * n_hidden2

        # Layer 3: Hidden2 → Output
        scale3 = math.sqrt(2.0 / n_hidden2)
        self.weights_ho = [random.gauss(0, scale3) for _ in range(n_hidden2)]
        self.bias_o = 0.0

        # Untuk menyimpan riwayat loss
        self.loss_history = []

    def forward(self, inputs):
        """
        Forward pass: menghitung output jaringan.
        
        Args:
            inputs: List fitur input [x1, x2, ..., xn]
            
        Returns:
            output: Nilai prediksi (0-1)
        """
        # Hidden Layer 1 (ReLU)
        self.hidden1_inputs = inputs
        self.hidden1_outputs = []
        for j in range(self.n_hidden1):
            total = self.bias_h1[j]
            for i in range(self.n_input):
                total += self.weights_ih1[j][i] * inputs[i]
            self.hidden1_outputs.append(relu(total))

        # Hidden Layer 2 (ReLU)
        self.hidden2_outputs = []
        for j in range(self.n_hidden2):
            total = self.bias_h2[j]
            for i in range(self.n_hidden1):
                total += self.weights_h1h2[j][i] * self.hidden1_outputs[i]
            self.hidden2_outputs.append(relu(total))

        # Output Layer (Sigmoid)
        total = self.bias_o
        for i in range(self.n_hidden2):
            total += self.weights_ho[i] * self.hidden2_outputs[i]
        self.output = sigmoid(total)

        return self.output

    def backward(self, inputs, target):
        """
        Backward pass: menghitung gradien dan update bobot.
        
        Args:
            inputs: List fitur input
            target: Label target (0 atau 1)
        """
        # ---- Output Layer Error ----
        output_error = self.output - target
        output_delta = output_error * sigmoid_derivative(self.output)

        # ---- Hidden Layer 2 Error ----
        hidden2_deltas = []
        for j in range(self.n_hidden2):
            error = output_delta * self.weights_ho[j]
            hidden2_deltas.append(error * relu_derivative(self.hidden2_outputs[j]))

        # ---- Hidden Layer 1 Error ----
        hidden1_deltas = []
        for j in range(self.n_hidden1):
            error = 0.0
            for k in range(self.n_hidden2):
                error += hidden2_deltas[k] * self.weights_h1h2[k][j]
            hidden1_deltas.append(error * relu_derivative(self.hidden1_outputs[j]))

        # ---- Update bobot Output Layer ----
        for i in range(self.n_hidden2):
            self.weights_ho[i] -= self.lr * output_delta * self.hidden2_outputs[i]
        self.bias_o -= self.lr * output_delta

        # ---- Update bobot Hidden Layer 2 ----
        for j in range(self.n_hidden2):
            for i in range(self.n_hidden1):
                self.weights_h1h2[j][i] -= self.lr * hidden2_deltas[j] * self.hidden1_outputs[i]
            self.bias_h2[j] -= self.lr * hidden2_deltas[j]

        # ---- Update bobot Hidden Layer 1 ----
        for j in range(self.n_hidden1):
            for i in range(self.n_input):
                self.weights_ih1[j][i] -= self.lr * hidden1_deltas[j] * inputs[i]
            self.bias_h1[j] -= self.lr * hidden1_deltas[j]

    def _save_weights(self):
        """Menyimpan snapshot bobot saat ini."""
        import copy
        return {
            'weights_ih1': copy.deepcopy(self.weights_ih1),
            'bias_h1': list(self.bias_h1),
            'weights_h1h2': copy.deepcopy(self.weights_h1h2),
            'bias_h2': list(self.bias_h2),
            'weights_ho': list(self.weights_ho),
            'bias_o': self.bias_o
        }

    def _load_weights(self, snapshot):
        """Memuat snapshot bobot."""
        import copy
        self.weights_ih1 = copy.deepcopy(snapshot['weights_ih1'])
        self.bias_h1 = list(snapshot['bias_h1'])
        self.weights_h1h2 = copy.deepcopy(snapshot['weights_h1h2'])
        self.bias_h2 = list(snapshot['bias_h2'])
        self.weights_ho = list(snapshot['weights_ho'])
        self.bias_o = snapshot['bias_o']

    def train(self, X_train, y_train, epochs=100, verbose=True,
              X_val=None, y_val=None, select_best_epoch=True,
              epoch_range=(50, 100)):
        """
        Melatih model ANN dengan pemilihan epoch terbaik.
        
        Args:
            X_train: Data training (list of lists)
            y_train: Label training (list)
            epochs: Jumlah epoch maksimum
            verbose: Cetak progress training
            X_val: Data validasi/testing untuk memilih epoch terbaik
            y_val: Label validasi/testing
            select_best_epoch: Jika True, pilih epoch dengan akurasi validasi terbaik
            epoch_range: Tuple (min_epoch, max_epoch) untuk rentang pencarian epoch terbaik
        """
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        best_val_acc = -1
        best_epoch = epochs
        best_weights = None
        min_epoch, max_epoch = epoch_range

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0

            # Shuffle data setiap epoch
            indices = list(range(len(X_train)))
            random.shuffle(indices)

            for idx in indices:
                inputs = X_train[idx]
                target = y_train[idx]

                # Forward pass
                output = self.forward(inputs)

                # Hitung Binary Cross-Entropy Loss
                epsilon = 1e-15
                output_clipped = max(epsilon, min(1 - epsilon, output))
                loss = -(target * math.log(output_clipped) + (1 - target) * math.log(1 - output_clipped))
                total_loss += loss

                # Hitung akurasi
                predicted = 1 if output >= 0.5 else 0
                if predicted == target:
                    correct += 1

                # Backward pass
                self.backward(inputs, target)

            avg_loss = total_loss / len(X_train)
            train_accuracy = correct / len(X_train) * 100
            self.loss_history.append(avg_loss)
            self.train_acc_history.append(train_accuracy)

            # Hitung akurasi validasi jika data tersedia
            val_accuracy = 0.0
            if X_val is not None and y_val is not None:
                val_preds, _ = self.predict_batch(X_val)
                val_correct = sum(1 for t, p in zip(y_val, val_preds) if t == p)
                val_accuracy = val_correct / len(y_val) * 100
                self.val_acc_history.append(val_accuracy)

                # Simpan model terbaik dalam rentang epoch yang ditentukan
                if select_best_epoch and min_epoch <= epoch <= max_epoch:
                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        best_epoch = epoch
                        best_weights = self._save_weights()

            if verbose and (epoch % 10 == 0 or epoch == 1):
                if X_val is not None:
                    marker = " <-- terbaik" if (epoch == best_epoch and select_best_epoch) else ""
                    print(f"  Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f} | "
                          f"Akurasi Train: {train_accuracy:.2f}% | "
                          f"Akurasi Val: {val_accuracy:.2f}%{marker}")
                else:
                    print(f"  Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f} | Akurasi Training: {train_accuracy:.2f}%")

        # Muat kembali bobot terbaik
        if select_best_epoch and best_weights is not None:
            self._load_weights(best_weights)
            self.best_epoch = best_epoch
            self.best_val_acc = best_val_acc
            if verbose:
                print(f"\n  >> Epoch terbaik: {best_epoch} (Akurasi Validasi: {best_val_acc:.2f}%)")
                print(f"  >> Bobot model dikembalikan ke epoch {best_epoch}")
        else:
            self.best_epoch = epochs
            self.best_val_acc = val_accuracy if X_val is not None else 0.0

    def predict(self, inputs):
        """Prediksi kelas untuk satu sampel input."""
        output = self.forward(inputs)
        return 1 if output >= 0.5 else 0, output

    def predict_batch(self, X):
        """Prediksi kelas untuk batch data."""
        predictions = []
        probabilities = []
        for inputs in X:
            pred, prob = self.predict(inputs)
            predictions.append(pred)
            probabilities.append(prob)
        return predictions, probabilities

    def save_to_file(self, filepath):
        """
        Menyimpan model (bobot, bias, arsitektur) ke file JSON.

        Args:
            filepath: Path file JSON tujuan penyimpanan.
        """
        import json
        model_data = {
            'architecture': {
                'n_input': self.n_input,
                'n_hidden1': self.n_hidden1,
                'n_hidden2': self.n_hidden2,
                'learning_rate': self.lr
            },
            'weights': {
                'weights_ih1': self.weights_ih1,
                'bias_h1': self.bias_h1,
                'weights_h1h2': self.weights_h1h2,
                'bias_h2': self.bias_h2,
                'weights_ho': self.weights_ho,
                'bias_o': self.bias_o
            },
            'training_info': {
                'best_epoch': getattr(self, 'best_epoch', None),
                'best_val_acc': getattr(self, 'best_val_acc', None)
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath):
        """
        Memuat model dari file JSON yang sudah disimpan.

        Args:
            filepath: Path file JSON model.

        Returns:
            Instance ANN dengan bobot yang sudah dimuat.
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        arch = model_data['architecture']
        # Buat instance baru tanpa random init yang akan di-overwrite
        model = cls(
            n_input=arch['n_input'],
            n_hidden1=arch['n_hidden1'],
            n_hidden2=arch['n_hidden2'],
            learning_rate=arch['learning_rate']
        )

        # Muat bobot dari file
        w = model_data['weights']
        model.weights_ih1 = w['weights_ih1']
        model.bias_h1 = w['bias_h1']
        model.weights_h1h2 = w['weights_h1h2']
        model.bias_h2 = w['bias_h2']
        model.weights_ho = w['weights_ho']
        model.bias_o = w['bias_o']

        # Muat info training
        info = model_data.get('training_info', {})
        model.best_epoch = info.get('best_epoch')
        model.best_val_acc = info.get('best_val_acc')

        return model

    def get_architecture_info(self):
        """Mengembalikan informasi arsitektur model."""
        total_params = (
            self.n_input * self.n_hidden1 + self.n_hidden1 +       # Layer 1
            self.n_hidden1 * self.n_hidden2 + self.n_hidden2 +     # Layer 2
            self.n_hidden2 + 1                                      # Output
        )
        return {
            'input': self.n_input,
            'hidden1': self.n_hidden1,
            'hidden2': self.n_hidden2,
            'output': 1,
            'total_params': total_params,
            'learning_rate': self.lr,
            'activation_hidden': 'ReLU',
            'activation_output': 'Sigmoid'
        }
