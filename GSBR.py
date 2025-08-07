import numpy as np
from typing import Literal

class BasicRegressor():
    """
    Gradient Supported Basic Regressor (GSBR) - Sebuah model regresi linear
    yang diimplementasikan dari nol menggunakan Batch Gradient Descent.

    Model ini mendukung regularisasi L1, L2, dan Elastic Net, serta fitur
    fit_intercept dan early stopping berbasis toleransi loss.

    Attributes:
        max_iter (int): Jumlah maksimum iterasi (epoch) untuk pelatihan.
        learning_rate (float): Tingkat pembelajaran untuk pembaruan bobot.
                                (Saat ini menggunakan fixed learning rate).
        verbose (int): Level verbosity. 0 untuk tidak ada output, 1 untuk mencetak
                       progress training per iterasi.
        penalty (Literal["L1", "L2", "elasticnet"] | None): Jenis regularisasi
                                                              yang digunakan.
                                                              'L1' untuk Lasso,
                                                              'L2' untuk Ridge,
                                                              'elasticnet' untuk
                                                              kombinasi keduanya,
                                                              atau None untuk tanpa
                                                              regularisasi.
        alpha (float): Kekuatan (magnitude) regularisasi. Semakin tinggi nilainya,
                       semakin besar penalti terhadap bobot yang besar.
        l1_ratio (float): Proporsi regularisasi L1 dalam Elastic Net.
                          Digunakan jika penalty='elasticnet'.
                          0.0 berarti murni L2, 1.0 berarti murni L1.
        fit_intercept (bool): Jika True, model akan menghitung bias/intercept.
                              Jika False, diasumsikan regresi melewati origin (0,0).
        tol (float): Toleransi untuk early stopping. Pelatihan berhenti jika
                     perubahan loss antara dua iterasi berturut-turut kurang
                     dari nilai ini.

        loss_history (list): List yang menyimpan nilai loss pada setiap iterasi.
        weights (np.ndarray): Bobot (koefisien) model setelah pelatihan.
        b (float): Bias (intercept) model setelah pelatihan.
        """
    def __init__(self, max_iter: int=100, learning_rate: float=0.01, verbose: int=0, penalty: Literal["l1", "l2", "elasticnet"] | None="l2", alpha: float=0.0001, l1_ratio: float=0.5, fit_intercept: bool=True, tol: float=0.0001, loss: Literal["mse", "rmse"] | None="mse"):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.intercept = fit_intercept
        self.tol = tol
        self.loss = loss
    
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha

        self.loss_history = []
        self.weights = None
        self.b = 0.0

    def mse(self, X, y) -> float:
        """
        Menghitung Mean Squared Error (MSE) dari prediksi model
        ditambah dengan istilah penalti regularisasi (jika ada).

        Args:
            X (np.ndarray): Fitur yang sudah diskalakan.
            y (np.ndarray): Target yang sudah diskalakan.

        Returns:
            float: Total nilai loss (MSE + penalty).
        """

        mse_cal = X @ self.weights + self.b - y
        mse = np.mean(mse_cal**2)

        penalty = 0

        if self.penalty == "l1":
          penalty = self.alpha * np.sum(np.abs(self.weights))

        elif self.penalty == "l2":
          penalty = self.alpha * np.sum(self.weights**2)

        elif self.penalty == "elasticnet":
          l1 = self.l1_ratio * np.sum(np.abs(self.weights))
          l2 = (1 - self.l1_ratio) * np.sum(self.weights**2)
          penalty = self.alpha * (l1 + l2)

        if self.loss == "rmse":
          mse = np.sqrt(mse)
           
        return mse + penalty
    
    def grad(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Menghitung gradien dari loss function (MSE ditambah penalti)
        terhadap bobot dan bias.

        Args:
            X (np.ndarray): Fitur yang sudah diskalakan.
            y (np.ndarray): Target yang sudah diskalakan.

        Returns:
            tuple[np.ndarray, float]: Tuple berisi gradien bobot (ndarray)
                                     dan gradien bias (float).
        """

        f = X @ self.weights + self.b - y

        grad_w = X.T @ (2 * f) / len(X)

        if self.intercept:
          grad_b = np.mean(2 * f)

        else:
          grad_b = 0.0

        grad_w_penalty = np.zeros_like(self.weights)

        if self.penalty == "l1":
            grad_w_penalty = self.alpha * np.sign(self.weights)

        elif self.penalty == "l2":
            grad_w_penalty = 2 * self.alpha * self.weights

        elif self.penalty == "elasticnet":
            l1 = self.l1_ratio * np.sign(self.weights)
            l2 = 2 * ((1 - self.l1_ratio) * self.weights)
            grad_w_penalty = self.alpha * (l2 + l1)

        else:
           raise ValueError("Invalid penalty type, Choose from 'l1', 'l2', or 'elasticnet'")

        grad_w = grad_w + grad_w_penalty

        return grad_w, grad_b
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Membuat prediksi menggunakan model yang sudah dilatih.

        Args:
            X_test (np.ndarray): Data fitur uji.

        Returns:
            np.ndarray: Array NumPy prediksi target dalam skala aslinya.

        Raises:
            ValueError: Jika model belum dilatih (bobot belum didefinisikan).
        """

        if X_test.ndim == 1:
            X_processed = X_test.reshape(-1, 1)

        else:
            X_processed = X_test

        if self.weights is None:
            raise ValueError("Weight not defined, try to train the model with fit() function first")

        pred = X_processed @ self.weights + self.b

        return pred

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Melatih model regresi menggunakan Batch Gradient Descent.

        Menginisialisasi bobot, mengiterasi untuk memperbarui bobot
        dan bias berdasarkan gradien loss function (MSE) ditambah penalti
        regularisasi (jika diterapkan).

        Pelatihan berhenti jika mencapai max_iter atau jika perubahan loss
        kurang dari nilai toleransi (tol).

        Args:
            X_train (np.ndarray): Data fitur pelatihan.
            y_train (np.ndarray): Data target pelatihan.
        """

        if X_train.ndim == 1:
            X_processed = X_train.reshape(-1, 1)

        else:
            X_processed = X_train

        num_samples, num_features = X_processed.shape

        if self.weights is None or self.weights.shape[0] != num_features:
         self.weights = np.zeros(num_features)
        
        if isinstance(y_train, (np.ndarray, list, tuple)):
            y_processed = np.asarray(y_train)

        else:
            y_processed = y_train.to_numpy()


        if not np.all(np.isfinite(X_processed)):
          raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        if not np.all(np.isfinite(y_processed)):
          raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")

        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )

        for i in range(self.max_iter):
            try:
              grad_w, grad_b = self.grad(X_processed, y_processed)
            except ValueError:
               grad_w, grad_b = self.grad(X_processed, y_processed.flatten())
            self.weights -= self.learning_rate * grad_w

            if self.intercept:
             self.b -= self.learning_rate * grad_b
            try:
              mse = self.mse(X_processed, y_processed)

            except ValueError:
               mse = self.mse(X_processed, y_processed.flatten())
            self.loss_history.append(mse)

            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                print(f"Warning: Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")
                break

            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.isnan(self.b) or np.isinf(self.b):
                    raise ValueError(f"There's NaN in epoch {i + 1} during the training process")

            
            if self.verbose == 1:
                print(f"{i + 1}. Grad_w: {np.mean(self.weights):.4f}, Grad_b: {self.b:.4f}, Loss: {mse:.4f}")
            
            try:
              if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                 break

            except IndexError:
                None