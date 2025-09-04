# GSBC – Klasifikasi Dasar Berbasis Gradien

## Gambaran Umum

GSBC (Gradient Supported Basic Classifier) adalah model klasifikasi buatan yang pake logistik regresi dengan optimasi gradien turun (*gradient descent*). Model ini bisa dipake buat klasifikasi biner maupun multi-kelas pake strategi *One-vs-Rest* (OvR).

## Instalasi & Kebutuhan

```bash
pip install numpy scipy
```

## Rumus Matematis

### Klasifikasi Biner

Model ini pake logistik regresi dengan fungsi sigmoid:

**Fungsi Sigmoid**: $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

**Fungsi Kerugian Entropi Silang Biner**: $$L = -\\frac{1}{N} \\sum\_{i=1}^{N} \\left\[ y_i \\log(p_i) + (1 - y_i) \\log(1 - p_i) \\right\]$$

**Gradien**: $$\\frac{\\partial L}{\\partial w} = \\frac{1}{N} X^T (p - y)$$ $$\\frac{\\partial L}{\\partial b} = \\frac{1}{N} \\sum (p - y)$$

### Regularisasi

- **L1 (Lasso)**: $\\alpha \\sum |w_i|$
- **L2 (Ridge)**: $\\alpha \\sum w_i^2$
- **Elastic Net**: $\\alpha \\left\[ l1_ratio \\cdot \\sum |w_i| + (1 - l1_ratio) \\cdot \\sum w_i^2 \\right\]$

## Fitur Utama

- ✅ Dukung klasifikasi biner dan multi-kelas
- ✅ Bisa handle matriks padat dan jarang (*dense/sparse*)
- ✅ Pake kerugian entropi silang biner
- ✅ Dioptimasi dengan gradien turun batch
- ✅ Ada fitur logging dan *early stopping*
- ✅ Dukung regularisasi L1, L2, dan Elastic Net
- ✅ Validasi input yang lengkap
- ✅ Strategi *One-vs-Rest* untuk masalah multi-kelas

## Parameter

| Parameter | Tipe | Default | Deskripsi |
| --- | --- | --- | --- |
| `max_iter` | `int` | `1000` | Jumlah maksimum iterasi gradien turun |
| `learning_rate` | `float` | `0.001` | Ukuran langkah untuk pembaruan gradien turun |
| `verbose` | `int` | `0` | Jika 1, tampilkan progres pelatihan (epoch, kerugian, dll.) |
| `fit_intercept` | `bool` | `True` | Apakah menyertakan terma bias (*intercept*) |
| `tol` | `float` | `0.0001` | Toleransi untuk *early stopping* berdasarkan konvergensi kerugian |
| `penalty` | `Literal['l1', 'l2', 'elasticnet'] | None` | `None` | Tipe regularisasi |
| `alpha` | `float` | `0.001` | Kekuatan regularisasi (digunakan jika *penalty* bukan None) |
| `l1_ratio` | `float` | `0.5` | Parameter campuran untuk *elastic net* (0 &lt;= l1_ratio &lt;= 1) |

## Atribut Model (Setelah Pelatihan)

| Atribut | Tipe | Deskripsi |
| --- | --- | --- |
| `weights` | `np.ndarray` | Bobot fitur yang dipelajari |
| `b` | `float` | Terma bias (*intercept*) |
| `loss_history` | `List[float]` | Nilai kerugian di setiap iterasi |
| `classes` | `np.ndarray` | Label kelas unik yang ditemukan saat pelatihan |
| `n_classes` | `int` | Jumlah kelas unik |
| `binary_classifiers` | `List[BasicClassifier]` | Klasifikasi OvR untuk masalah multi-kelas |

## Referensi API

### `BasicClassifier.__init__()`

Inisialisasi klasifikasi dengan hiperparameter.

### `BasicClassifier.fit(X_train, y_train)`

Latih klasifikasi dengan data yang diberikan.

**Parameter**:

- `X_train`: Fitur pelatihan (n_samples, n_features), padat atau jarang
- `y_train`: Label target pelatihan (n_samples,)

**Pengecualian**:

- `ValueError`: Jika data input mengandung NaN/Inf atau bentuk tidak sesuai

### `BasicClassifier.predict_proba(X_test)`

Prediksi probabilitas kelas untuk data uji.

**Hasil**:

- Untuk biner: (n_samples,) probabilitas
- Untuk multi-kelas: (n_samples, n_classes) matriks probabilitas

### `BasicClassifier.predict(X_test)`

Prediksi label kelas untuk data uji.

**Hasil**: (n_samples,) label kelas yang diprediksi

### `BasicClassifier.score(X_test, y_test)`

Hitung skor akurasi klasifikasi.

**Hasil**: Skor akurasi antara 0 dan 1

## Contoh Penggunaan

### Klasifikasi Biner

```python
from BasicModels.GSBC import BasicClassifier
import numpy as np

# Bikin data contoh
X = np.random.randn(100, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Inisialisasi dan latih model
model = BasicClassifier(
    max_iter=500, 
    learning_rate=0.01, 
    verbose=1,
    penalty='l2',
    alpha=0.1
)
model.fit(X, y)

# Bikin prediksi
preds = model.predict(X)
probas = model.predict_proba(X)
accuracy = model.score(X, y)

print(f"Akurasi: {accuracy:.4f}")
print(f"Kerugian akhir: {model.loss_history[-1]:.6f}")
```

### Klasifikasi Multi-Kelas

```python
from BasicModels.GSBC import BasicClassifier
from sklearn.datasets import make_classification
import numpy as np

# Bikin data multi-kelas
X, y = make_classification(
    n_samples=200, 
    n_features=4, 
    n_classes=3, 
    n_informative=3,
    random_state=42
)

# Latih dengan regularisasi
model = BasicClassifier(
    max_iter=1000,
    learning_rate=0.005,
    verbose=1,
    penalty='elasticnet',
    alpha=0.01,
    l1_ratio=0.3
)
model.fit(X, y)

# Prediksi dan evaluasi
preds = model.predict(X)
accuracy = np.mean(preds == y)
print(f"Akurasi multi-kelas: {accuracy:.4f}")
print(f"Kelas: {model.classes}")
```

### Dengan Matriks Jarang

```python
from BasicModels.GSBC import BasicClassifier
from scipy.sparse import csr_matrix
import numpy as np

# Bikin data jarang
X_sparse = csr_matrix(np.random.randn(100, 10))
y = (np.random.rand(100) > 0.5).astype(int)

model = BasicClassifier(max_iter=300, learning_rate=0.02)
model.fit(X_sparse, y)

print("Pelatihan matriks jarang selesai!")
```

## Praktik Terbaik

### Penyetelan Hiperparameter

- Mulai dengan `learning_rate=0.01` dan sesuaikan berdasarkan konvergensi
- Pakai `max_iter=1000-5000` untuk masalah yang lebih rumit
- Setel `tol=1e-4` untuk *early stopping* yang masuk akal
- Untuk regularisasi: mulai dengan nilai `alpha` kecil (0.001-0.1)

### Pra-pemrosesan Data

- Skalakan fitur ke rata-rata=0, standar deviasi=1 untuk konvergensi lebih baik
- Tangani nilai yang hilang sebelum pelatihan
- Untuk masalah multi-kelas, pastikan kelas seimbang atau pakai bobot kelas

### Pantau Pelatihan

```python
# Pantau progres pelatihan
import matplotlib.pyplot as plt

model = BasicClassifier(max_iter=1000, learning_rate=0.01, verbose=1)
model.fit(X_train, y_train)

plt.plot(model.loss_history)
plt.xlabel('Iterasi')
plt.ylabel('Kerugian')
plt.title('Konvergensi Kerugian Pelatihan')
plt.show()
```

## Penanganan Kesalahan

Model ini udah dilengkapi pengecekan kesalahan yang lengkap:

- Validasi input untuk nilai NaN/Inf
- Pengecekan konsistensi bentuk data
- Validasi parameter (nilai positif, rentang valid)
- *Early stopping* untuk ketidakstabilan numerik

## Pertimbangan Performa

- **Memori**: Efisien untuk matriks padat dan jarang
- **Kecepatan**: Gradien turun batch cocok untuk dataset ukuran sedang
- **Skalabilitas**: Untuk dataset super besar, pertimbangkan varian mini-batch

## Perbandingan dengan scikit-learn

| Fitur | GSBC | scikit-learn LogisticRegression |
| --- | --- | --- |
| Regularisasi kustom | ✅ | ✅ |
| Dukungan matriks jarang | ✅ | ✅ |
| Multi-kelas OvR | ✅ | ✅ |
| *Early stopping* | ✅ | ✅ |
| Penjadwalan *learning rate* | ❌ | ✅ |
| Pilihan solver | ❌ | ✅ |
| Bobot kelas | ❌ | ✅ |

## Lisensi

Implementasi ini disediakan sebagai bagian dari paket BasicModels.