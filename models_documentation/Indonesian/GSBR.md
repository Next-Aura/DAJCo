# GSBR – Regresor Dasar Berbasis Gradien

## Gambaran Umum
GSBR (Gradient Supported Basic Regressor) adalah model regresi linear buatan yang dukung berbagai teknik regularisasi kayak L1, L2, dan ElasticNet. Model ini pake *Mean Squared Error* (MSE) atau *Root Mean Squared Error* (RMSE) sebagai fungsi kerugian, dioptimasi pake gradien turun (*gradient descent*).

## Instalasi & Kebutuhan
```bash
pip install numpy scipy scikit-learn
```

## Rumus Matematis

### Regresi Linear
Model ini pake regresi linear dengan fungsi prediksi:

**Prediksi**: 
$$\hat{y} = Xw + b$$

**Mean Squared Error (MSE)**:
$$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE)**:
$$L_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

### Regularisasi
- **L1 (Lasso)**: $\alpha \sum |w_i|$
- **L2 (Ridge)**: $\alpha \sum w_i^2$
- **Elastic Net**: $\alpha \left[ l1\_ratio \cdot \sum |w_i| + (1 - l1\_ratio) \cdot \sum w_i^2 \right]$

### Gradien
**Gradien Bobot**:
$$\frac{\partial L}{\partial w} = \frac{2}{N} X^T (Xw + b - y) + \text{gradien regularisasi}$$

**Gradien Bias**:
$$\frac{\partial L}{\partial b} = \frac{2}{N} \sum (Xw + b - y)$$

## Fitur Utama
- ✅ Dukung regularisasi L1, L2, dan ElasticNet
- ✅ Bisa handle input matriks padat dan jarang
- ✅ Fungsi kerugian: MSE atau RMSE
- ✅ Ada *early stopping* dan output pelatihan verbose
- ✅ Validasi input yang lengkap
- ✅ Optimasi gradien turun dengan kontrol *learning rate*

## Parameter

| Parameter | Tipe | Default | Deskripsi |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `100` | Jumlah maksimum iterasi gradien turun |
| `learning_rate` | `float` | `0.01` | Ukuran langkah untuk pembaruan gradien turun |
| `verbose` | `int` | `0` | Jika 1, tampilkan progres pelatihan (epoch, bobot, bias, kerugian) |
| `penalty` | `Literal['l1', 'l2', 'elasticnet'] \| None` | `'l2'` | Tipe regularisasi |
| `alpha` | `float` | `0.0001` | Kekuatan regularisasi |
| `l1_ratio` | `float` | `0.5` | Parameter campuran untuk ElasticNet (0 = L2, 1 = L1) |
| `fit_intercept` | `bool` | `True` | Apakah menyertakan terma bias (*intercept*) |
| `tol` | `float` | `0.0001` | Toleransi untuk *early stopping* berdasarkan konvergensi kerugian |
| `loss` | `Literal['mse', 'rmse']` | `'mse'` | Fungsi kerugian yang diminimalkan |

## Atribut Model (Setelah Pelatihan)

| Atribut | Tipe | Deskripsi |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Bobot fitur yang dipelajari |
| `b` | `float` | Terma bias (*intercept*) |
| `loss_history` | `List[float]` | Nilai kerugian di setiap iterasi |

## Referensi API

### `BasicRegressor.__init__()`
Inisialisasi regresor dengan hiperparameter.

### `BasicRegressor.fit(X_train, y_train)`
Latih regresor dengan data yang diberikan.

**Parameter**:
- `X_train`: Fitur pelatihan (n_samples, n_features), padat atau jarang
- `y_train`: Nilai target pelatihan (n_samples,)

**Pengecualian**:
- `ValueError`: Jika data input mengandung NaN/Inf atau bentuk tidak sesuai
- `TypeError`: Jika data input mengandung tipe non-numerik

### `BasicRegressor.predict(X_test)`
Prediksi nilai target untuk data uji.

**Hasil**: (n_samples,) nilai yang diprediksi

**Pengecualian**:
- `ValueError`: Jika model belum dilatih

### `BasicRegressor.loss_score()`
Dapatkan rata-rata kerugian dari semua iterasi.

**Hasil**: Nilai kerugian rata-rata

**Pengecualian**:
- `ValueError`: Jika riwayat pelatihan tidak tersedia

## Contoh Penggunaan

### Regresi Dasar dengan Regularisasi L2
```python
from BasicModels.GSBR import BasicRegressor
import numpy as np
from sklearn.datasets import make_regression

# Bikin data contoh
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Inisialisasi dan latih model
model = BasicRegressor(
    max_iter=1000,
    learning_rate=0.01,
    verbose=1,
    penalty='l2',
    alpha=0.1,
    fit_intercept=True,
    tol=1e-4,
    loss='mse'
)
model.fit(X, y)

# Bikin prediksi
preds = model.predict(X)
print(f"Bobot akhir: {model.weights}")
print(f"Bias akhir: {model.b}")
print(f"Rata-rata kerugian: {model.loss_score():.6f}")
```

### Regularisasi ElasticNet dengan Kerugian RMSE
```python
from BasicModels.GSBR import BasicRegressor
from sklearn.preprocessing import StandardScaler

# Skalakan fitur untuk konvergensi lebih baik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = BasicRegressor(
    max_iter=2000,
    learning_rate=0.005,
    penalty='elasticnet',
    alpha=0.05,
    l1_ratio=0.3,
    loss='rmse',
    verbose=1
)
model.fit(X_scaled, y)

print(f"Model ElasticNet dilatih dengan RMSE: {model.loss_score():.4f}")
```

### Dukungan Matriks Jarang
```python
from BasicModels.GSBR import BasicRegressor
from scipy.sparse import csr_matrix
import numpy as np

# Bikin data jarang
X_sparse = csr_matrix(np.random.randn(100, 10))
y = np.random.randn(100)

model = BasicRegressor(
    max_iter=500,
    learning_rate=0.02,
    verbose=0
)
model.fit(X_sparse, y)

print("Pelatihan matriks jarang selesai dengan sukses!")
```

### Tanpa Regularisasi (Ordinary Least Squares)
```python
model = BasicRegressor(
    max_iter=1000,
    learning_rate=0.01,
    penalty=None,
    fit_intercept=True,
    verbose=1
)
model.fit(X, y)

print(f"Model OLS - Kerugian akhir: {model.loss_history[-1]:.6f}")
```

## Praktik Terbaik

### Penyetelan Hiperparameter
- Mulai dengan `learning_rate=0.01-0.1` dan sesuaikan berdasarkan konvergensi
- Pakai `max_iter=1000-5000` untuk masalah yang lebih rumit
- Setel `tol=1e-4` untuk *early stopping* yang masuk akal
- Untuk regularisasi: mulai dengan nilai `alpha` kecil (0.001-0.1)
- Untuk ElasticNet: `l1_ratio=0.5` memberikan campuran seimbang L1/L2

### Pra-pemrosesan Data
- Skalakan fitur ke rata-rata=0, standar deviasi=1 untuk konvergensi lebih baik
- Tangani nilai yang hilang sebelum pelatihan
- Pertimbangkan *feature engineering* untuk performa lebih baik

### Pantau Pelatihan
```python
import matplotlib.pyplot as plt

model = BasicRegressor(max_iter=1000, learning_rate=0.01, verbose=0)
model.fit(X_train, y_train)

plt.plot(model.loss_history)
plt.xlabel('Iterasi')
plt.ylabel('Kerugian')
plt.title('Konvergensi Kerugian Pelatihan')
plt.grid(True)
plt.show()
```

## Penanganan Kesalahan

Model ini udah dilengkapi pengecekan kesalahan yang lengkap:
- Validasi input untuk nilai NaN/Inf
- Pengecekan tipe untuk data numerik
- Pengecekan konsistensi bentuk data
- Validasi parameter (nilai positif, rentang valid)
- *Early stopping* untuk ketidakstabilan numerik

## Pertimbangan Performa

- **Memori**: Efisien untuk matriks padat dan jarang
- **Kecepatan**: Gradien turun batch cocok untuk dataset ukuran sedang
- **Skalabilitas**: Untuk dataset super besar, pertimbangkan varian mini-batch

## Perbandingan dengan scikit-learn

| Fitur | GSBR | scikit-learn LinearRegression/ElasticNet |
|---------|------|------------------------------------------|
| Regularisasi kustom | ✅ | ✅ |
| Dukungan matriks jarang | ✅ | ✅ |
| Gradien turun | ✅ | ❌ (pake solusi analitis) |
| *Early stopping* | ✅ | ✅ |
| Kontrol *learning rate* | ✅ | ❌ |
| Pilihan solver ganda | ❌ | ✅ |
| Validasi silang | ❌ | ✅ |

## Lisensi

Implementasi ini disediakan sebagai bagian dari paket BasicModels.