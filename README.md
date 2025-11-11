# Analisis Sentimen Timnas

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis sentimen publik terhadap isu "Indonesia Gagal Lolos Piala Dunia" melalui komentar yang dikumpulkan dari YouTube. Dengan menggunakan model Support Vector Machine (SVM), proyek ini mengklasifikasikan sentimen komentar menjadi positif, negatif, atau netral.

## Struktur Proyek
Proyek ini memiliki struktur sebagai berikut:

```
analisis-sentimen-timnas
├── src
│   ├── data_collector.py       # Kelas untuk mengumpulkan data dari YouTube
│   ├── preprocessor.py          # Kelas untuk preprocessing teks komentar
│   ├── model.py                 # Kelas untuk model SVM
│   ├── config.py                # Konfigurasi pengaturan proyek
│   └── utils
│       └── helpers.py           # Fungsi utilitas untuk analisis data
├── notebooks
│   ├── analisis_sentimen_complete.ipynb  # Notebook analisis lengkap
│   └── exploratory_data_analysis.ipynb    # Notebook untuk analisis eksploratori
├── data
│   ├── raw
│   │   ├── comments_real.csv    # Komentar mentah dari YouTube
│   │   └── videos_real.csv      # Data video mentah dari YouTube
│   └── processed
│       ├── final_processed_dataset_real.csv  # Dataset yang telah diproses
│       └── sample_predictions_real.csv       # Prediksi dari model
├── models
│   ├── sentiment_svm_model_real_data.pkl     # Model SVM yang telah dilatih
│   ├── tfidf_vectorizer_real_data.pkl        # Vektorizer TF-IDF
│   ├── label_encoder_real_data.pkl            # Encoder label untuk sentimen
│   └── model_performance_metrics.json         # Metode performa model
├── results
│   ├── visualizations        # Direktori untuk menyimpan visualisasi
│   └── reports              # Direktori untuk menyimpan laporan
├── requirements.txt         # Daftar dependensi Python
├── config.yaml              # Pengaturan konfigurasi dalam format YAML
└── README.md                # Dokumentasi proyek
```

## Instalasi
1. Clone repositori ini ke mesin lokal Anda.
2. Install dependensi yang diperlukan dengan menjalankan:
   ```
   pip install -r requirements.txt
   ```

## Penggunaan
1. Jalankan notebook `analisis_sentimen_complete.ipynb` untuk melakukan analisis lengkap.
2. Gunakan notebook `exploratory_data_analysis.ipynb` untuk melakukan analisis eksploratori terhadap data komentar.

## Kontribusi
Kontribusi sangat diterima! Silakan buka issue atau pull request untuk berkontribusi pada proyek ini.

## Lisensi
Proyek ini dilisensikan di bawah MIT License. Silakan lihat file LICENSE untuk informasi lebih lanjut.