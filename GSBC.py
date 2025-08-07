import numpy as np
from typing import Literal
from scipy.sparse import issparse, spmatrix

class BasicClassifier:
    def __init__(self, max_iter: int=1000, learning_rate: float=0.001, verbose: int=0, fit_intercept: bool=True, tol: float=0.0001):
        self.max_iter = max_iter
        self.lr_rate = learning_rate
        self.intercept = fit_intercept
        self.verbose = verbose
        self.tol = tol # Toleransi untuk early stopping dasar

        self.weights = None
        self.b = 0.0
        self.loss_history= []
        
        # Atribut untuk multi-class
        self.classes = None
        self.n_classes = 0
        self.binary_classifiers = [] # Diubah namanya dari 'classifier' agar lebih jelas untuk OvR

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def binary_ce(self, y_true, y_pred_proba):
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

        loss = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        return np.mean(loss)
    
    def grad(self, X_scaled: np.ndarray | spmatrix,  y_true: np.ndarray) -> tuple[np.ndarray, float]:
        # Logika gradien sudah disatukan dan berlaku untuk sparse atau dense
        z = X_scaled @ self.weights
        y_pred_proba = self.sigmoid(z)

        error = y_pred_proba - y_true

        grad_w = (X_scaled.T @ error) / X_scaled.shape[0] # Menggunakan X_scaled.shape[0] untuk jumlah sampel
        grad_b = np.mean(error)

        return grad_w, grad_b
    
    def fit(self, X_train, y_train):
        # --- Pre-processing Input X ---
        if not issparse(X_train): # Mengecek apakah sparse atau tidak di awal
            if X_train.ndim == 1:
                X_processed = X_train.reshape(-1, 1)
            else:
                X_processed = X_train
            X_processed = np.asarray(X_processed) # Pastikan NumPy array
        else:
            X_processed = X_train # Biarkan sparse jika memang sparse

        num_samples, num_features = X_processed.shape

        # --- Pre-processing Input y ---
        y_processed = np.asarray(y_train).flatten() # Pastikan y selalu 1D NumPy array

        # --- Validasi Data ---
        # Perbaikan: Lebih spesifik dalam menangani TypeError atau hilangkan jika tidak diperlukan
        if issparse(X_processed):
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values in its data. Please clean your data.")
        else:
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")
        
        if not np.all(np.isfinite(y_processed)):
            raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")

        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )

        # --- Identifikasi Kelas Unik ---
        self.classes = np.unique(y_processed)
        self.n_classes = len(self.classes)
        self.loss_history = [] # Reset loss history setiap kali fit dipanggil

        # --- Logika Pelatihan ---
        if self.n_classes == 2:
            # === Klasifikasi Biner ===
            if self.weights is None or self.weights.shape[0] != num_features:
                self.weights = np.zeros(num_features)
            self.b = 0.0 # Reset bias

            for i in range(self.max_iter):
                grad_w, grad_b = self.grad(X_processed, y_processed)

                self.weights -= self.lr_rate * grad_w
                if self.intercept:
                    self.b -= self.lr_rate * grad_b

                z_current = X_processed @ self.weights
                if self.intercept:
                    z_current += self.b

                y_proba_current = self.sigmoid(z_current)
                loss = self.binary_ce(y_processed, y_proba_current)
                self.loss_history.append(loss)

                # Cek NaN/Inf
                if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                    print(f"Warning: Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")
                    break
                if not np.isfinite(loss):
                    print(f"Warning: Loss became NaN/Inf at epoch {i + 1}. Stopping training early.")
                    break

                # Early Stopping
                if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    print(f"Early stopping at epoch {i+1}: Loss change ({abs(self.loss_history[-1] - self.loss_history[-2]):.6f}) below tolerance ({self.tol:.6f}).")
                    break

                # Perbaikan: Pindahkan verbose ke dalam loop
                if self.verbose == 1:
                    print(f"Epoch {i+1}/{self.max_iter} - Loss: {loss:.6f}")

        elif self.n_classes > 2:
            # === Klasifikasi Multi-Class dengan OvR ===
            self.binary_classifiers = [] # Reset list classifier
            print(f"Training {self.n_classes} binary classifiers using One-vs-Rest (OvR) strategy...")
            
            # Loop untuk setiap kelas unik
            for class_label in self.classes:
                if self.verbose == 1: # Print detail training OvR jika verbose
                    print(f"  Training classifier for class {class_label} vs all others...")
                
                # Buat label biner untuk OvR: 1 jika sampel adalah class_label, 0 jika bukan
                y_ovr = (y_processed == class_label).astype(int)
                
                # Buat instance baru dari BasicClassifier untuk setiap OvR
                # Penting: Setiap OvR classifier harus punya hyperparameter sendiri
                # Kita salin hyperparameter dari main classifier, verbose untuk OvR internal kita set 0 agar tidak terlalu ramai
                clf = BasicClassifier(max_iter=self.max_iter, 
                                      learning_rate=self.lr_rate, 
                                      verbose=0, # Atur verbose internal menjadi 0
                                      fit_intercept=self.intercept, 
                                      tol=self.tol) # Menggunakan tol yang sama
                
                # Latih classifier biner ini
                clf.fit(X_processed, y_ovr) # X_processed di sini adalah data asli

                self.binary_classifiers.append(clf)
        else:
           raise ValueError("Class label must have at least 2 types.")

    def predict_proba(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        # --- Pre-processing Input X_test ---
        if not issparse(X_test):
            if X_test.ndim == 1:
                X_processed = X_test.reshape(-1, 1)
            else:
                X_processed = X_test
            X_processed = np.asarray(X_processed)
        else:
            X_processed = X_test

        if self.n_classes == 0: # Cek apakah model sudah dilatih
             raise ValueError("Model not trained. Call fit() first.")
        
        # --- Logika Prediksi Probabilitas ---
        if self.n_classes == 2:
            # === Prediksi Biner ===
            # Pastikan weights dan b ada jika model biner yang dilatih
            if self.weights is None:
                raise ValueError("Weight is none, try to fit the model with dataset first before predicting")

            z = X_processed @ self.weights
            if self.intercept:
               z += self.b
            proba = self.sigmoid(z)
            return proba
        
        elif self.n_classes > 2:
            # === Prediksi Multi-Class dengan OvR ===
            if not self.binary_classifiers:
                raise ValueError("OvR classifiers not trained. Call fit() first for multi-class data.")
            
            # Perbaikan: Syntax np.zeros()
            all_probas = np.zeros((X_processed.shape[0], self.n_classes))

            # Untuk setiap classifier biner yang sudah dilatih
            for i, clf in enumerate(self.binary_classifiers):
                # Dapatkan probabilitas kelas 1 (probabilitas untuk kelas yang dilatih)
                class_proba = clf.predict_proba(X_processed)
                all_probas[:, i] = class_proba
            
            # Perbaikan: Pindahkan return ke luar loop
            return all_probas
        else:
           raise ValueError("Target label must have at least 2 types")
      
    def predict(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        # Dapatkan probabilitas terlebih dahulu
        probas = self.predict_proba(X_test) # Panggil predict_proba untuk mengurus X_test processing

        if self.n_classes == 2:
            pred_class = (probas >= 0.5).astype(int)
        elif self.n_classes > 2:
            pred_class = np.argmax(probas, axis=1)

            # Perbaikan: Cek self.classes dan len(self.classes)
            if self.classes is not None and len(self.classes) == self.n_classes:
                # Mapping kembali indeks ke label kelas asli jika kelasnya bukan 0, 1, 2...
                pred_class = np.array([self.classes[idx] for idx in pred_class])
        else:
            raise ValueError("Model is not trained or invalid number of class.")
        
        return pred_class