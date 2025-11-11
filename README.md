# Analisis Sentimen Timnas Indonesia - Multi-Layer Analytics

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis sentimen publik terhadap isu "Indonesia Gagal Lolos Piala Dunia" melalui komentar yang dikumpulkan dari YouTube. Dengan menggunakan model Support Vector Machine (SVM), proyek ini tidak hanya mengklasifikasikan sentimen (positif, negatif, netral), tetapi juga menyediakan **analisis berlapis (layered analytics)** yang mencakup:

- 🎭 **Emotion Analysis**: Deteksi emosi multi-label (marah, kecewa, sedih, senang, bangga, takut)
- 🎯 **Aspect Detection**: Identifikasi aspek/topik yang dibahas (manajemen, pelatih, pemain, strategi, wasit, PSSI, federasi, fanbase)
- ⚠️ **Toxicity Detection**: Deteksi ujaran kasar/offensive
- 💭 **Stance Classification**: Klasifikasi posisi (pro, kontra, tidak_jelas)
- 📝 **Intent Classification**: Klasifikasi tujuan komentar (pertanyaan, komplain, saran, ajakan, humor, informasi)

## Struktur Proyek
Proyek ini memiliki struktur sebagai berikut:

```
analisis-sentimen-timnas
├── src
│   ├── data_collector.py       # Kelas untuk mengumpulkan data dari YouTube
│   ├── preprocessor.py          # Kelas untuk preprocessing teks komentar
│   ├── model.py                 # Kelas untuk model SVM sentimen
│   ├── layered_classifier.py    # Kelas untuk multi-layer classifiers (NEW!)
│   ├── config.py                # Konfigurasi pengaturan proyek
│   └── utils
│       ├── helpers.py           # Fungsi utilitas untuk analisis data
│       └── layered_utils.py     # Utilitas untuk layered analytics (NEW!)
├── notebooks
│   ├── analisis_sentimen_complete.ipynb  # Notebook analisis lengkap
│   ├── exploratory_data_analysis.ipynb    # Notebook EDA
│   └── layered_analytics_demo.ipynb       # Demo layered analytics (NEW!)
├── data
│   ├── raw
│   │   ├── comments_real.csv    # Komentar mentah dari YouTube
│   │   └── videos_real.csv      # Data video mentah
│   ├── processed
│   │   ├── final_processed_dataset_real.csv  # Dataset terproses
│   │   └── sample_predictions_real.csv       # Prediksi model
│   └── labelled                 # Data berlabel manual (NEW!)
│       ├── LABELING_GUIDE.md    # Panduan pelabelan
│       └── labelled_comments_template.csv  # Template labeling
├── models
│   ├── sentiment_svm_model_real_data.pkl     # Model SVM sentimen
│   ├── tfidf_vectorizer_real_data.pkl        # Vectorizer TF-IDF
│   ├── label_encoder_real_data.pkl            # Label encoder sentimen
│   ├── emotion_svm.pkl          # Model emosi (jika sudah trained)
│   ├── aspect_svm.pkl           # Model aspek (jika sudah trained)
│   ├── toxicity_svm.pkl         # Model toxicity (jika sudah trained)
│   ├── stance_svm.pkl           # Model stance (jika sudah trained)
│   ├── intent_svm.pkl           # Model intent (jika sudah trained)
│   └── model_performance_metrics.json         # Metrik performa
├── results
│   ├── visualizations        # Visualisasi analisis
│   └── reports              # Laporan hasil
├── requirements.txt         # Dependensi Python
├── config.yaml              # Konfigurasi YAML (termasuk layer config)
└── README.md                # Dokumentasi proyek
```

## Instalasi
1. Clone repositori ini ke mesin lokal Anda.
2. Install dependensi yang diperlukan dengan menjalankan:
   ```
   pip install -r requirements.txt
   ```

## Penggunaan

### Basic Sentiment Analysis
1. Jalankan notebook `analisis_sentimen_complete.ipynb` untuk melakukan analisis sentimen dasar.
2. Gunakan notebook `exploratory_data_analysis.ipynb` untuk EDA terhadap data komentar.

### Multi-Layer Analytics (Advanced)
Untuk analisis yang lebih mendalam dengan layer tambahan:

1. **Persiapan Data Berlabel**
   - Lihat panduan di `data/labelled/LABELING_GUIDE.md`
   - Labeli subset data (minimal 300-500 per kelas untuk setiap layer)
   - Simpan sebagai CSV di folder `data/labelled/`

2. **Training Layer Models**
   - Buka notebook `layered_analytics_demo.ipynb`
   - Ikuti langkah-langkah training untuk setiap layer:
     - Emotion Classifier (multi-label)
     - Aspect Classifier (multi-label)
     - Toxicity Classifier
     - Stance Classifier
     - Intent Classifier

3. **Inference dan Analisis**
   - Jalankan prediksi berlapis pada komentar baru
   - Ekspor hasil ke CSV dengan semua layer predictions
   - Visualisasi distribusi per layer

### Output yang Dihasilkan

Setelah menjalankan layered analytics, Anda akan mendapatkan:

**CSV Output** (`data/processed/layered_predictions.csv`):
```csv
text,sentiment,sentiment_confidence,emotions,aspects,toxicity_label,toxicity_score,stance,intent,low_confidence_flag
"Pelatih harus diganti!",negatif,0.89,"marah,kecewa","pelatih,strategi",non-toxic,0.12,kontra,komplain,False
"Kapan Indonesia bisa lolos?",netral,0.72,sedih,,non-toxic,0.08,tidak_jelas,pertanyaan,False
```

**Metrik Performa** (`models/metrics_[layer].json`):
- Accuracy, Precision, Recall, F1-score per layer
- Confusion matrix (untuk single-label)
- Hamming loss, Jaccard score (untuk multi-label)
- Per-label performance breakdown

**Visualisasi**:
- Distribusi label per layer
- Wordclouds kata dominan per emosi/aspek
- Trend temporal sentimen & emosi
- Confusion matrices

## Konfigurasi Layered Analytics

Edit `config.yaml` untuk mengaktifkan/menonaktifkan layer:

```yaml
layered_analytics:
  enable_emotion: true
  enable_aspect: true
  enable_toxicity: true
  enable_stance: true
  enable_intent: true
  
  low_confidence_threshold: 0.60
  toxicity_threshold: 0.50
```

## Cara Menggunakan Model untuk Prediksi

```python
from src.layered_classifier import EmotionClassifier, AspectClassifier
from src.utils.layered_utils import create_layered_output_dataframe
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize classifiers
emotion_clf = EmotionClassifier(
    emotion_labels=config['layered_analytics']['emotion_labels']
)
emotion_clf.load('models/emotion_svm.pkl', 
                 'models/emotion_vectorizer.pkl',
                 'models/emotion_mlb.pkl')

# Predict
texts = ["Kecewa dengan performa tim", "Mantap, terus semangat!"]
emotion_labels, emotion_conf = emotion_clf.predict(texts)

print(emotion_labels)
# Output: [['kecewa'], ['senang', 'bangga']]
```

## Metodologi

### Pipeline Dasar (Sentimen)
1. Koleksi data YouTube → 2. Preprocessing → 3. TF-IDF → 4. SVM Training → 5. Evaluasi

### Pipeline Berlapis (Multi-Layer)
1. Sentimen (positif/netral/negatif)
2. Emosi (multi-label: marah, kecewa, sedih, senang, bangga, takut)
3. Aspek (multi-label: manajemen, pelatih, pemain, strategi, wasit, PSSI, dll)
4. Toxicity (toxic/non-toxic)
5. Stance (pro/kontra/tidak_jelas)
6. Intent (pertanyaan/komplain/saran/ajakan/humor/informasi)

Setiap layer menggunakan classifier terpisah (SVM/LinearSVC) yang di-train secara independen.

## Metrik Evaluasi

### Single-Label Layers (Sentiment, Toxicity, Stance, Intent)
- Accuracy
- Precision, Recall, F1-Score (macro)
- Confusion Matrix
- Per-class metrics

### Multi-Label Layers (Emotion, Aspect)
- Macro Precision, Recall, F1-Score
- Hamming Loss (semakin rendah semakin baik)
- Jaccard Score (semakin tinggi semakin baik)
- Per-label support dan F1

## Use Cases

### 1. Monitoring Sentimen Real-time
Gunakan model untuk mengklasifikasikan komentar baru secara otomatis dan deteksi lonjakan sentimen negatif.

### 2. Prioritisasi Respon
Filter komentar berdasarkan toxicity dan intent untuk menentukan mana yang perlu dijawab/ditindaklanjuti.

### 3. Analisis Aspek Mendalam
Identifikasi aspek mana (pelatih, pemain, strategi, dll) yang paling banyak dikritik atau dipuji.

### 4. Segmentasi Emosi
Pahami nuansa emosi di balik sentimen untuk komunikasi yang lebih empatis.

### 5. Dashboard Analytics
Integrasikan output ke dashboard untuk visualisasi real-time distribusi sentimen, emosi, dan aspek.

## Limitasi & Pengembangan Lanjutan

### Limitasi Saat Ini
- Model baseline menggunakan TF-IDF (tidak menangkap konteks semantik)
- Belum menangani sarkasme dengan baik
- Multi-label threshold fixed (belum di-tune per label)
- Bahasa Indonesia informal/campur kode masih menantang

### Roadmap Pengembangan
- [ ] Implementasi word embeddings (Word2Vec/FastText Indo)
- [ ] Eksplorasi transformer-based models (IndoBERT, mBERT)
- [ ] Threshold tuning per label (GridSearch)
- [ ] Topic modeling tambahan (LDA/BERTopic)
- [ ] Dashboard interaktif (Streamlit/Dash)
- [ ] API endpoint untuk real-time prediction
- [ ] Active learning untuk continuous improvement

## Requirements Tambahan untuk Layered Analytics

Tambahkan ke `requirements.txt` jika belum ada:
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
joblib>=1.0.0
```

## Kontribusi
Kontribusi sangat diterima! Silakan buka issue atau pull request untuk berkontribusi pada proyek ini.

## Lisensi
Proyek ini dilisensikan di bawah MIT License. Silakan lihat file LICENSE untuk informasi lebih lanjut.