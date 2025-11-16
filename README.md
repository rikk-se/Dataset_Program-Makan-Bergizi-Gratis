# Dataset: Analisis Sentimen Kebijakan Makan Bergizi Gratis

Repositori ini berisi dataset yang digunakan untuk penelitian "Analisis Sentimen Masyarakat Terhadap Kebijakan Makan Bergizi Gratis Menggunakan Hybrid Lexicon-Based & Machine Learning".

Dataset ini dikumpulkan dari platform media sosial X (sebelumnya Twitter) pada periode Oktober-November 2024 dan mencakup total **12.389 tweet** publik berbahasa Indonesia.

Fokus dari repositori ini adalah untuk menyediakan dataset yang sudah bersih dan terlabeli, siap untuk digunakan dalam pemodelan *machine learning*.

## 📂 File Dataset

Kami menyediakan dua file CSV utama untuk transparansi proses:

1.  `preprocessed_data.csv`
2.  `labeled_data_hybrid.csv`

---

### 1. `preprocessed_data.csv`

File ini berisi data tweet yang telah melalui seluruh tahapan preprocessing, namun **sebelum** proses labeling. File ini berguna jika Anda ingin menguji metode labeling Anda sendiri atau melakukan analisis teks yang tidak terstruktur.

**Struktur Kolom Utama:**

| Nama Kolom | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `reply_text` | string | Teks tweet asli (mentah) dari hasil scraping. |
| `clean_text` | string | Teks tweet yang telah melalui 4 tahap preprocessing (case folding, cleansing, normalisasi slang, dan stopword removal). |
| `...` | - | (Kolom-kolom metadata lain dari proses scraping) |

---

### 2. `labeled_data_hybrid.csv`

File ini adalah dataset **final** yang digunakan untuk melatih model *machine learning* dalam penelitian kami. File ini berisi data dari `preprocessed_data.csv` yang telah ditambahi dengan label sentimen yang dihasilkan oleh metode *hybrid*.

**Struktur Kolom Utama:**

| Nama Kolom | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `clean_text` | string | Teks bersih yang siap digunakan sebagai fitur. |
| **`sentiment_label`** | **string** | **Label sentimen final (hasil hybrid). Nilainya: `positif`, `negatif`, atau `netral`.** |
| `confidence` | float | Skor kepercayaan (0.0 - 1.0) dari proses pelabelan. |
| `lexicon_sentiment`| string | Prediksi sentimen murni dari metode Lexicon-Based. |
| `lexicon_score` | float | Skor numerik mentah yang dihasilkan oleh metode Lexicon-Based. |
| `bert_sentiment` | string | Prediksi sentimen murni dari model IndoBERT (`w11wo/indonesian-roberta...`). |
| `labeling_method` | string | Metode yang digunakan untuk menentukan label final (cth: `agreement`, `bert_priority`, `weighted_voting`). |

---

## ⚙️ Metodologi Singkat Pembuatan Dataset

### 1. Preprocessing

Teks bersih (`clean_text`) dihasilkan melalui 4 tahap:

1.  **Case Folding:** Mengubah seluruh teks menjadi huruf kecil.
2.  **Cleansing:** Menghapus URL, *mention* (@username), *hashtag* (#), emoji, tanda baca, dan karakter numerik.
3.  **Slang Normalization:** Mengubah kata-kata tidak baku (cth: `gak`, `bgt`, `dgn`) menjadi kata baku (`tidak`, `banget`, `dengan`) menggunakan kamus slang kustom.
4.  **Stopword Removal:** Menghapus kata-kata umum (cth: `yang`, `di`, `ini`) menggunakan daftar Sastrawi, namun **mempertahankan** kata-kata negasi (cth: `tidak`, `belum`, `jangan`) dan *intensifiers* (cth: `sangat`, `kurang`) yang penting untuk analisis sentimen.

### 2. Hybrid Labeling

Label sentimen (`sentiment_label`) dihasilkan **tanpa pelabelan manual**. Kami menggunakan metode *hybrid* otomatis yang menggabungkan:

* **Lexicon-Based:** Menggunakan kamus sentimen kustom yang diperkaya dengan *domain-specific terms* (cth: `korupsi`, `bergizi`, `higienis`) dan aturan negasi.
* **IndoBERT:** Menggunakan model *transformer* `w11wo/indonesian-roberta-base-sentiment-classifier` untuk prediksi berbasis konteks.

Label final ditentukan melalui logika *weighted voting* dan prioritas berdasarkan skor kepercayaan dari kedua metode tersebut.

## Python

```python
import pandas as pd

# Memuat dataset yang sudah dilabeli (file utama)
df = pd.read_csv('labeled_data_hybrid.csv')

# Melihat 5 baris pertama
print(df[['clean_text', 'sentiment_label']].head())

# Melihat distribusi sentimen
print("\nDistribusi Sentimen:")
print(df['sentiment_label'].value_counts())
