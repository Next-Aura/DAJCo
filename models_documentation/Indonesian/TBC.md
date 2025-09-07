# TBC – Tree Backend Classifier

## Gambaran Umum
TBC (Tree Backend Classifier) adalah implementasi klasifikasi pohon keputusan buatan sendiri yang mendukung berbagai kriteria ketidakmurnian seperti Gini impurity, entropy, dan log loss. Model ini menggunakan pembagian biner rekursif untuk membangun pohon klasifikasi dan termasuk dukungan untuk matriks sparse dengan kompilasi Numba JIT opsional untuk performa.

## Instalasi & Kebutuhan
```bash
pip install numpy scipy numba
```

## Rumus Matematis

### Klasifikasi Pohon Keputusan
Model ini membangun pohon biner dengan membagi ruang fitur secara rekursif:

**Prediksi**:
$$\hat{y} = \arg\max_c \left( \frac{1}{|L|} \sum_{i \in L} \mathbb{I}(y_i = c) \right)$$

Di mana $L$ adalah himpunan sampel pelatihan di node daun dan $c$ mewakili label kelas.

### Kriteria Ketidakmurnian

**Gini Impurity**:
$$I_{Gini} = 1 - \sum_{c=1}^{C} p_c^2$$

Di mana $p_c$ adalah proporsi sampel kelas $c$ di node.

**Entropy**:
$$I_{Entropy} = -\sum_{c=1}^{C} p_c \log_2(p_c)$$

**Log Loss (Cross-Entropy)**:
$$I_{LogLoss} = -\sum_{c=1}^{C} p_c \log(p_c)$$

### Kualitas Pembagian
**Penurunan Ketidakmurnian (Gain)**:
$$Gain = I_{parent} - \frac{|S_{left}|}{|S|} I_{left} - \frac{|S_{right}|}{|S|} I_{right}$$

## Fitur Utama
- ✅ Mendukung kriteria Gini, Entropy, dan Log Loss
- ✅ Bekerja dengan input matriks padat dan sparse
- ✅ Kompilasi Numba JIT opsional untuk performa
- ✅ Validasi input lengkap dengan peringatan
- ✅ Pembangunan pohon rekursif dengan kriteria pemberhentian yang dapat dikonfigurasi
- ✅ Dukungan klasifikasi multi-kelas
- ✅ Pemilihan fitur berbasis varians untuk matriks sparse

## Parameter

| Parameter | Tipe | Default | Deskripsi |
|-----------|------|---------|-------------|
| `max_depth` | `int \| None` | `2` | Kedalaman maksimum pohon. Jika None atau <=0, default ke 2 |
| `min_samples_leaf` | `int \| None` | `5` | Jumlah minimum sampel yang diperlukan di node daun |
| `criterion` | `Literal['gini', 'entropy', 'log_loss'] \| None` | `'gini'` | Fungsi untuk mengukur kualitas pembagian |
| `min_samples_split` | `int \| None` | `2` | Jumlah minimum sampel yang diperlukan untuk membagi node internal |
| `min_impurity_decrease` | `float \| None` | `0.0` | Penurunan ketidakmurnian minimum yang diperlukan untuk pembagian |

## Atribut Model (Setelah Pelatihan)

| Atribut | Tipe | Deskripsi |
|-----------|------|-------------|
| `tree` | `dict` | Kamus bersarang yang mewakili struktur pohon |
| `max_depth` | `int` | Kedalaman maksimum pohon |
| `min_samples_leaf` | `int` | Sampel minimum per daun |
| `criterion` | `str` | Kriteria ketidakmurnian yang digunakan |
| `min_samples_split` | `int` | Sampel minimum yang diperlukan untuk membagi |
| `min_impurity_decrease` | `float` | Penurunan ketidakmurnian minimum untuk pembagian |

## Referensi API

### `TBClassifier.__init__()`
Inisialisasi klasifikasi pohon dengan hiperparameter.

### `TBClassifier.fit(X, y)`
Bangun klasifikasi pohon keputusan dari set pelatihan.

**Parameter**:
- `X`: Fitur pelatihan (n_samples, n_features), padat atau sparse
- `y`: Label kelas target (n_samples,)

**Mengembalikan**: self

**Membuang**:
- `ValueError`: Jika X dan y panjangnya tidak cocok atau berisi data tidak valid

### `TBClassifier.predict(X)`
Prediksi label kelas untuk X.

**Parameter**:
- `X`: Fitur uji (n_samples, n_features), padat atau sparse

**Mengembalikan**: (n_samples,) label kelas yang diprediksi

## Contoh Penggunaan

### Klasifikasi Pohon Keputusan Dasar
```python
from BasicModels.TBC import TBClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Buat data sampel
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model
model = TBClassifier(
    max_depth=5,
    min_samples_leaf=10,
    criterion='gini',
    min_samples_split=20
)
model.fit(X_train, y_train)

# Buat prediksi
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluasi
train_accuracy = np.mean(train_preds == y_train)
test_accuracy = np.mean(test_preds == y_test)

print(f"Akurasi Latih: {train_accuracy:.4f}")
print(f"Akurasi Uji: {test_accuracy:.4f}")
```

### Menggunakan Kriteria Berbeda
```python
from BasicModels.TBC import TBClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)

# Kriteria Gini
model_gini = TBClassifier(criterion='gini', max_depth=4)
model_gini.fit(X, y)
print(f"Akurasi Gini: {np.mean(model_gini.predict(X) == y):.4f}")

# Kriteria Entropy
model_entropy = TBClassifier(criterion='entropy', max_depth=4)
model_entropy.fit(X, y)
print(f"Akurasi Entropy: {np.mean(model_entropy.predict(X) == y):.4f}")

# Kriteria Log Loss
model_logloss = TBClassifier(criterion='log_loss', max_depth=4)
model_logloss.fit(X, y)
print(f"Akurasi Log Loss: {np.mean(model_logloss.predict(X) == y):.4f}")
```

### Klasifikasi Multi-Kelas
```python
from BasicModels.TBC import TBClassifier
from sklearn.datasets import make_classification
import numpy as np

# Hasilkan data multi-kelas
X, y = make_classification(
    n_samples=800,
    n_features=8,
    n_classes=4,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

model = TBClassifier(
    max_depth=6,
    min_samples_leaf=15,
    criterion='entropy'
)
model.fit(X, y)

predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Akurasi Multi-kelas: {accuracy:.4f}")
print(f"Kelas Unik: {np.unique(y)}")
```

### Dukungan Matriks Sparse
```python
from BasicModels.TBC import TBClassifier
from scipy.sparse import csr_matrix
import numpy as np

# Buat data sparse
X_dense = np.random.randn(300, 15)
X_sparse = csr_matrix(X_dense)
y = (X_dense[:, 0] + X_dense[:, 1] > 0).astype(int)

model = TBClassifier(max_depth=4, criterion='gini')
model.fit(X_sparse, y)

print("Pelatihan matriks sparse selesai dengan sukses!")
accuracy = np.mean(model.predict(X_sparse) == y)
print(f"Akurasi: {accuracy:.4f}")
```

### Menangani Peringatan Parameter
```python
from BasicModels.TBC import TBClassifier
import warnings

# Ini akan memicu peringatan dan menggunakan default
model = TBClassifier(
    max_depth=-1,  # Tidak valid, akan peringatkan dan set ke 2
    min_samples_leaf=0,  # Tidak valid, akan peringatkan dan set ke 5
    criterion='gini'
)

# Peringatan akan ditampilkan:
# "max_depth harus bilangan bulat positif. Mengatur ke nilai default 2."
# "min_samples_leaf harus bilangan bulat positif. Mengatur ke nilai default 5."

print(f"max_depth Aktual: {model.max_depth}")
print(f"min_samples_leaf Aktual: {model.min_samples_leaf}")
```

## Praktik Terbaik

### Penyetelan Hiperparameter
- Mulai dengan `max_depth=3-5` dan tingkatkan untuk dataset kompleks
- Gunakan `min_samples_leaf=5-10` untuk mencegah overfitting
- Untuk dataset besar, tingkatkan `min_samples_split` ke 20-50
- Setel `min_impurity_decrease > 0` untuk memangkas pembagian lemah
- Pilih kriteria berdasarkan dataset: 'gini' untuk kecepatan, 'entropy' untuk gain informasi

### Pra-pemrosesan Data
- Tidak perlu scaling untuk pohon keputusan
- Tangani fitur kategorikal dengan encoding ke numerik
- Pertimbangkan feature engineering untuk pembagian lebih baik
- Hapus atau imputasi nilai yang hilang
- Pastikan label kelas adalah bilangan bulat mulai dari 0

### Visualisasi Pohon
```python
# Inspeksi struktur pohon sederhana
def print_tree(node, depth=0):
    indent = "  " * depth
    if "label" in node:
        print(f"{indent}Daun: Kelas {node['label']}")
    else:
        print(f"{indent}Pembagian pada fitur {node['feature']} <= {node['value']:.4f}")
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)

print_tree(model.tree)
```

## Penanganan Kesalahan

Model ini menyertakan pengecekan kesalahan yang lengkap:
- Validasi input untuk array kosong
- Pengecekan konsistensi bentuk antara X dan y
- Validasi parameter dengan peringatan informatif
- Validasi kriteria
- Pengecekan kompatibilitas matriks sparse
- Penanganan kasus tepi (kelas tunggal, tidak ada pembagian valid)

## Pertimbangan Performa

- **Memori**: Efisien dengan matriks padat dan sparse
- **Kecepatan**: Akselerasi Numba JIT untuk kalkulasi ketidakmurnian
- **Skalabilitas**: Cocok untuk dataset hingga 100k sampel
- **Multi-kelas**: Menangani beberapa kelas secara efisien

## Perbandingan dengan scikit-learn

| Fitur | TBC | scikit-learn DecisionTreeClassifier |
|---------|------|-------------------------------------|
| Kriteria kustom | ✅ | ✅ |
| Dukungan matriks sparse | ✅ | ✅ |
| Akselerasi Numba | ✅ | ❌ |
| Peringatan parameter | ✅ | ❌ |
| Pemangkasan pohon | ✅ | ✅ |
| Pemangkasan kompleksitas biaya | ❌ | ✅ |
| Multi-output | ❌ | ✅ |
| Bobot kelas | ❌ | ✅ |

## Lisensi

Implementasi ini disediakan sebagai bagian dari paket BasicModels.
