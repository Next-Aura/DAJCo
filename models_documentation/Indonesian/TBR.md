# TBR – Tree Backend Regressor

## Gambaran Umum
TBR (Tree Backend Regressor) adalah implementasi regresor pohon keputusan buatan sendiri yang mendukung berbagai kriteria ketidakmurnian seperti MSE, Friedman MSE, Absolute Error, dan Poisson deviance. Model ini menggunakan pembagian biner rekursif untuk membangun pohon regresi dan termasuk dukungan untuk matriks sparse dengan kompilasi Numba JIT opsional untuk performa.

## Instalasi & Kebutuhan
```bash
pip install numpy scipy numba
```

## Rumus Matematis

### Regresi Pohon Keputusan
Model ini membangun pohon biner dengan membagi ruang fitur secara rekursif:

**Prediksi**:
$$\hat{y} = \frac{1}{|L|} \sum_{i \in L} y_i$$

Di mana $L$ adalah himpunan sampel pelatihan di node daun.

### Kriteria Ketidakmurnian

**Mean Squared Error (MSE)**:
$$I_{MSE} = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

**Friedman MSE**:
$$I_{Friedman} = \frac{1}{|S|-1} \sum_{i \in S} (y_i - \bar{y})^2$$

**Mean Absolute Error**:
$$I_{MAE} = \frac{1}{|S|} \sum_{i \in S} |y_i - \bar{y}|$$

**Poisson Deviance**:
$$I_{Poisson} = 2 \sum_{i \in S} \left[ y_i \log\left(\frac{y_i}{\bar{y}}\right) - (y_i - \bar{y}) \right]$$

### Kualitas Pembagian
**Penurunan Ketidakmurnian (Gain)**:
$$Gain = I_{parent} - \frac{|S_{left}|}{|S|} I_{left} - \frac{|S_{right}|}{|S|} I_{right}$$

## Fitur Utama
- ✅ Mendukung MSE, Friedman MSE, Absolute Error, dan Poisson criteria
- ✅ Bekerja dengan input matriks padat dan sparse
- ✅ Kompilasi Numba JIT opsional untuk performa
- ✅ Validasi input lengkap
- ✅ Pembangunan pohon rekursif dengan kriteria pemberhentian yang dapat dikonfigurasi
- ✅ Dukungan pemilihan fitur
- ✅ Perhitungan skor R² untuk evaluasi

## Parameter

| Parameter | Tipe | Default | Deskripsi |
|-----------|------|---------|-------------|
| `max_depth` | `Optional[int]` | `5` | Kedalaman maksimum pohon. Jika None, node akan diperluas sampai semua daun murni |
| `min_samples_leaf` | `Optional[int]` | `1` | Jumlah minimum sampel yang diperlukan di node daun |
| `criterion` | `Literal['mse', 'friedman_mse', 'absolute_error', 'poisson']` | `'mse'` | Fungsi untuk mengukur kualitas pembagian |
| `max_features` | `Optional[int]` | `None` | Jumlah fitur untuk dipertimbangkan saat mencari pembagian terbaik |
| `random_state` | `Optional[int]` | `None` | Seed yang digunakan oleh generator angka acak |
| `min_samples_split` | `Optional[int]` | `2` | Jumlah minimum sampel yang diperlukan untuk membagi node internal |
| `min_impurity_decrease` | `Optional[float]` | `0.0` | Penurunan ketidakmurnian minimum yang diperlukan untuk pembagian |

## Atribut Model (Setelah Pelatihan)

| Atribut | Tipe | Deskripsi |
|-----------|------|-------------|
| `tree` | `dict` | Kamus bersarang yang mewakili struktur pohon |
| `max_depth` | `int` | Kedalaman maksimum pohon |
| `min_samples_leaf` | `int` | Sampel minimum per daun |
| `criterion` | `str` | Kriteria ketidakmurnian yang digunakan |
| `max_features` | `Optional[int]` | Jumlah fitur yang dipertimbangkan untuk pembagian |
| `random_state` | `Optional[int]` | State acak untuk reproduktibilitas |
| `min_samples_split` | `int` | Sampel minimum yang diperlukan untuk membagi |
| `min_impurity_decrease` | `float` | Penurunan ketidakmurnian minimum untuk pembagian |

## Referensi API

### `TBRegressor.__init__()`
Inisialisasi regresor pohon dengan hiperparameter.

### `TBRegressor.fit(X, y)`
Bangun regresor pohon keputusan dari set pelatihan.

**Parameter**:
- `X`: Fitur pelatihan (n_samples, n_features), padat atau sparse
- `y`: Nilai target (n_samples,)

**Mengembalikan**: self

**Membuang**:
- `ValueError`: Jika X dan y panjangnya tidak cocok atau berisi data tidak valid

### `TBRegressor.predict(X)`
Prediksi nilai target untuk X.

**Parameter**:
- `X`: Fitur uji (n_samples, n_features), padat atau sparse

**Mengembalikan**: (n_samples,) nilai yang diprediksi

### `TBRegressor.score(X, y)`
Mengembalikan koefisien determinasi R² dari prediksi.

**Parameter**:
- `X`: Fitur uji (n_samples, n_features)
- `y`: Nilai target yang benar (n_samples,)

**Mengembalikan**: Skor R² (float)

## Contoh Penggunaan

### Regresi Pohon Keputusan Dasar
```python
from BasicModels.TBR import TBRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Buat data sampel
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model
model = TBRegressor(
    max_depth=10,
    min_samples_leaf=5,
    criterion='mse',
    random_state=42
)
model.fit(X_train, y_train)

# Buat prediksi
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluasi
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"R² Latih: {train_score:.4f}")
print(f"R² Uji: {test_score:.4f}")
```

### Menggunakan Kriteria Berbeda
```python
from BasicModels.TBR import TBRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)

# Kriteria MSE
model_mse = TBRegressor(criterion='mse', max_depth=8)
model_mse.fit(X, y)
print(f"R² MSE: {model_mse.score(X, y):.4f}")

# Friedman MSE
model_friedman = TBRegressor(criterion='friedman_mse', max_depth=8)
model_friedman.fit(X, y)
print(f"R² Friedman MSE: {model_friedman.score(X, y):.4f}")

# Absolute Error
model_mae = TBRegressor(criterion='absolute_error', max_depth=8)
model_mae.fit(X, y)
print(f"R² MAE: {model_mae.score(X, y):.4f}")
```

### Kriteria Poisson untuk Data Count
```python
from BasicModels.TBR import TBRegressor
import numpy as np

# Hasilkan data mirip Poisson
np.random.seed(42)
X = np.random.randn(300, 4)
y = np.random.poisson(np.exp(X[:, 0] + 0.5 * X[:, 1]))  # Target Poisson

model = TBRegressor(
    criterion='poisson',
    max_depth=6,
    min_samples_leaf=10
)
model.fit(X, y)

print(f"Deviance Poisson R²: {model.score(X, y):.4f}")
```

### Dukungan Matriks Sparse
```python
from BasicModels.TBR import TBRegressor
from scipy.sparse import csr_matrix
import numpy as np

# Buat data sparse
X_dense = np.random.randn(200, 20)
X_sparse = csr_matrix(X_dense)
y = X_dense[:, 0] + 0.5 * X_dense[:, 1] + np.random.randn(200) * 0.1

model = TBRegressor(max_depth=8, random_state=42)
model.fit(X_sparse, y)

print("Pelatihan matriks sparse selesai dengan sukses!")
print(f"Skor R²: {model.score(X_sparse, y):.4f}")
```

### Pemilihan Fitur
```python
from BasicModels.TBR import TBRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Gunakan hanya 10 fitur untuk setiap pembagian
model = TBRegressor(
    max_features=10,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

print(f"R² mirip random forest: {model.score(X, y):.4f}")
```

## Praktik Terbaik

### Penyetelan Hiperparameter
- Mulai dengan `max_depth=5-10` dan tingkatkan untuk dataset kompleks
- Gunakan `min_samples_leaf=1-5` untuk mencegah overfitting
- Untuk dataset besar, tingkatkan `min_samples_split` ke 10-20
- Setel `min_impurity_decrease > 0` untuk memangkas pembagian lemah
- Gunakan `max_features` untuk perilaku mirip ensemble

### Pra-pemrosesan Data
- Tidak perlu scaling untuk pohon keputusan
- Tangani fitur kategorikal dengan encoding ke numerik
- Pertimbangkan feature engineering untuk pembagian lebih baik
- Hapus atau imputasi nilai yang hilang

### Visualisasi Pohon
```python
# Inspeksi struktur pohon sederhana
def print_tree(node, depth=0):
    indent = "  " * depth
    if "value" in node:
        print(f"{indent}Daun: {node['value']:.4f}")
    else:
        print(f"{indent}Pembagian pada fitur {node['feature']} <= {node['threshold']:.4f}")
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)

print_tree(model.tree)
```

## Penanganan Kesalahan

Model ini menyertakan pengecekan kesalahan yang lengkap:
- Validasi input untuk nilai NaN/Inf
- Pengecekan konsistensi bentuk antara X dan y
- Validasi parameter (nilai positif, rentang valid)
- Validasi kriteria
- Pengecekan kompatibilitas matriks sparse

## Pertimbangan Performa

- **Memori**: Efisien dengan matriks padat dan sparse
- **Kecepatan**: Akselerasi Numba JIT untuk kalkulasi varians
- **Skalabilitas**: Cocok untuk dataset hingga 100k sampel
- **Paralelisasi**: Pemilihan fitur dapat diparalelisasi

## Perbandingan dengan scikit-learn

| Fitur | TBR | scikit-learn DecisionTreeRegressor |
|---------|------|------------------------------------|
| Kriteria kustom | ✅ | ✅ |
| Dukungan matriks sparse | ✅ | ✅ |
| Kriteria Poisson | ✅ | ❌ |
| Akselerasi Numba | ✅ | ❌ |
| Pemilihan fitur | ✅ | ✅ |
| Pemangkasan pohon | ✅ | ✅ |
| Pemangkasan kompleksitas biaya | ❌ | ✅ |
| Multi-output | ❌ | ✅ |

## Lisensi

Implementasi ini disediakan sebagai bagian dari paket BasicModels.
