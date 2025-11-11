# 📊 Summary: Multi-Layer Analytics System

## ✅ Apa yang Sudah Dibangun

### 1. **Five Layer Classifiers** (5 Classifier Berlapis)

| Layer | Type | Classes/Labels | Purpose |
|-------|------|----------------|---------|
| **Emotion** | Multi-label | marah, kecewa, sedih, senang, bangga, takut | Deteksi nuansa emosi |
| **Aspect** | Multi-label | manajemen, pelatih, pemain, strategi, wasit, PSSI, federasi, fanbase | Identifikasi topik pembahasan |
| **Toxicity** | Binary | toxic, non-toxic | Deteksi ujaran kasar/offensive |
| **Stance** | 3-class | pro, kontra, tidak_jelas | Klasifikasi posisi opini |
| **Intent** | 6-class | pertanyaan, komplain, saran, ajakan, humor, informasi | Klasifikasi tujuan komentar |

### 2. **Complete Infrastructure**

✅ **Source Code**:
- `src/layered_classifier.py` - Semua classifier classes
- `src/utils/layered_utils.py` - Helper functions untuk evaluasi dan reporting

✅ **Configuration**:
- `config.yaml` - Enable/disable layers, thresholds, label definitions

✅ **Documentation**:
- `README.md` - Updated dengan layered analytics
- `QUICKSTART_LAYERED.md` - Quick start guide
- `data/labelled/LABELING_GUIDE.md` - Panduan pelabelan detail

✅ **Demo & Templates**:
- `notebooks/layered_analytics_demo.ipynb` - Complete demo notebook
- `data/labelled/labelled_comments_template.csv` - Template labeling

✅ **Git Repository**:
- Semua code di-commit dan di-push ke GitHub
- Repo: https://github.com/Fahri-Hilm/Analisis-SVM-Timnas

---

## 🎯 Output yang Dihasilkan

### Per Komentar:
```python
{
  "text": "Pelatih harus diganti, strateginya kacau!",
  "sentiment": "negatif",
  "sentiment_confidence": 0.89,
  "emotions": ["marah", "kecewa"],
  "aspects": ["pelatih", "strategi"],
  "toxicity_label": "non-toxic",
  "toxicity_score": 0.12,
  "stance": "kontra",
  "intent": "komplain",
  "low_confidence_flag": false
}
```

### Aggregate Analytics:
- Distribusi label per layer
- Top keywords/phrases per emotion/aspect
- Temporal trends (sentimen + emosi over time)
- Engagement patterns (like count per sentimen/emosi)
- Channel/video analysis

### Metrics per Layer:
- **Single-label**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Multi-label**: Macro F1, Hamming Loss, Jaccard Score, per-label support

---

## 📈 Use Cases Praktis

### 1. **Real-time Monitoring Dashboard**
```
├─ Sentiment Gauge (positif/netral/negatif %)
├─ Emotion Heatmap (marah↑, kecewa↑, senang↓)
├─ Top Aspects Being Discussed (pelatih 45%, strategi 32%, ...)
├─ Toxicity Alert (toxic spike detected!)
└─ Intent Distribution (komplain 40%, pertanyaan 25%, ...)
```

### 2. **Prioritisasi Respon**
```python
# Filter komentar yang perlu ditindaklanjuti
priority_comments = df[
    (df['intent'] == 'komplain') & 
    (df['toxicity_label'] == 'non-toxic') &
    (df['sentiment'] == 'negatif')
]
# → Fokus jawab komplain yang konstruktif
```

### 3. **Analisis Aspek Mendalam**
```python
# Aspek mana yang paling banyak dikritik?
negative_df = df[df['sentiment'] == 'negatif']
aspect_counts = negative_df['aspects'].str.split(',').explode().value_counts()
# → "pelatih" 120x, "strategi" 95x, "manajemen" 87x
```

### 4. **Segmentasi Emosi untuk Komunikasi**
```python
# Buat pesan khusus untuk setiap segmen emosi
kecewa_segment = df[df['emotions'].str.contains('kecewa')]
marah_segment = df[df['emotions'].str.contains('marah')]
# → Komunikasi empati untuk "kecewa", address concern untuk "marah"
```

### 5. **Early Warning System**
```python
# Deteksi lonjakan sentimen negatif + emosi marah
if (negative_pct > 60) and (emotion_marah_count > threshold):
    alert("⚠️ Sentiment crisis detected!")
```

---

## 🔄 Workflow: Dari Data Mentah → Insight

```
1. Koleksi Komentar YouTube
   └─> data/raw/comments_real.csv

2. Preprocessing (existing)
   └─> cleaning, normalisasi, tokenisasi
   └─> data/processed/final_processed_dataset_real.csv

3. **Manual Labeling** (NEW step)
   └─> Labeli subset (300-500 per class/label)
   └─> data/labelled/labelled_comments.csv

4. **Train Layered Models** (NEW)
   └─> Run: notebooks/layered_analytics_demo.ipynb
   └─> Output: models/*.pkl + metrics_*.json

5. **Inference pada Semua Komentar** (NEW)
   └─> Apply models pada full dataset
   └─> Output: data/processed/layered_predictions.csv

6. **Analisis & Visualisasi** (ENHANCED)
   ├─> Sentiment distribution (existing)
   ├─> Emotion heatmap (NEW)
   ├─> Aspect breakdown (NEW)
   ├─> Toxicity trend (NEW)
   ├─> Stance polarization (NEW)
   └─> Intent distribution (NEW)

7. **Report & Insight** (COMPREHENSIVE)
   └─> results/reports/layered_analytics_report.txt
   └─> Actionable recommendations
```

---

## 💡 Key Advantages

### Dibanding Sentimen 3-Kelas Saja:

| Aspek | Sentimen Saja | + Layered Analytics |
|-------|---------------|---------------------|
| **Granularity** | positif/netral/negatif | + 6 emosi, 8 aspek, toxicity, stance, intent |
| **Actionability** | "40% negatif" | "40% negatif → 25% kecewa pelatih, 15% marah strategi" |
| **Prioritisasi** | Semua negatif sama | Bisa filter: komplain konstruktif vs toxic rant |
| **Komunikasi** | Generic response | Tailored: address pelatih issue, explain strategi |
| **Monitoring** | Sentiment trend | Multi-dimensional: emosi+aspek+toxicity trend |

---

## 📊 Metrik Evaluasi yang Digunakan

### Single-Label Layers (Sentiment, Toxicity, Stance, Intent)
```python
{
  "accuracy": 0.85,
  "precision": 0.83,  # macro average
  "recall": 0.84,
  "f1_score": 0.83,
  "confusion_matrix": [[...], [...], [...]]
}
```

### Multi-Label Layers (Emotion, Aspect)
```python
{
  "macro_precision": 0.78,
  "macro_recall": 0.76,
  "macro_f1": 0.77,
  "hamming_loss": 0.12,      # lower is better
  "jaccard_score": 0.68,     # higher is better
  "per_label_f1": {
    "marah": 0.82,
    "kecewa": 0.79,
    ...
  }
}
```

---

## 🚀 Next Steps & Roadmap

### Immediate (1-2 minggu)
- [x] ✅ Build infrastructure
- [ ] Label 300-500 samples per class
- [ ] Train all layers
- [ ] Validate on test set
- [ ] Generate baseline report

### Short-term (1 bulan)
- [ ] Integrate dengan existing pipeline
- [ ] Create Streamlit dashboard prototype
- [ ] A/B test: sentimen saja vs layered
- [ ] Collect feedback dari stakeholder

### Mid-term (2-3 bulan)
- [ ] Expand training data (1000+ per class)
- [ ] Fine-tune thresholds per label
- [ ] Experiment dengan embeddings (Word2Vec Indo)
- [ ] Build API endpoint untuk real-time prediction

### Long-term (6+ bulan)
- [ ] Explore transformer models (IndoBERT)
- [ ] Active learning pipeline
- [ ] Real-time monitoring dashboard (production)
- [ ] Automated reporting & alerting
- [ ] Integration dengan CRM/ticketing system

---

## 📦 Deliverables Checklist

### Code & Infrastructure
- [x] 5 Classifier classes (emotion, aspect, toxicity, stance, intent)
- [x] Utilities untuk evaluation & reporting
- [x] Configuration system (config.yaml)
- [x] Demo notebook dengan contoh lengkap
- [x] Git repository dengan commit history

### Documentation
- [x] README updated dengan layered analytics
- [x] Quick start guide (QUICKSTART_LAYERED.md)
- [x] Labeling guide dengan definisi & contoh
- [x] Template CSV untuk labeling
- [x] Inline code documentation (docstrings)

### Output Examples
- [x] Demo predictions (dummy data)
- [x] Metrics JSON format examples
- [x] CSV output format specification
- [ ] Sample visualizations (need labeled data first)
- [ ] Sample comprehensive report (need labeled data first)

### Deployment Ready
- [x] Modular architecture (easy to extend)
- [x] Config-driven (easy to customize)
- [x] Reproducible (saved models + metrics)
- [x] Version controlled (GitHub)
- [ ] Production pipeline (need integration)

---

## 🎓 Cara Presentasi

### Slide 1: Problem Statement
"Sentimen 3-kelas tidak cukup untuk actionable insight"

### Slide 2: Solution - Multi-Layer Analytics
Diagram 5 layers dengan ikon

### Slide 3: Architecture
```
Komentar → Preprocessing → [Layer 1: Sentiment]
                          → [Layer 2: Emotion]
                          → [Layer 3: Aspect]
                          → [Layer 4: Toxicity]
                          → [Layer 5: Stance]
                          → [Layer 6: Intent]
                          → Comprehensive Output
```

### Slide 4: Output Example
Tunjukkan 1 komentar dengan semua layer predictions

### Slide 5: Use Case - Prioritisasi Respon
Filter komplain konstruktif

### Slide 6: Use Case - Analisis Aspek
"Pelatih" dikritik 45%, "Strategi" 32%

### Slide 7: Metrics & Validation
Tabel F1-score per layer

### Slide 8: Demo
Live demo notebook (jika waktu ada)

### Slide 9: Roadmap
Timeline short/mid/long-term

### Slide 10: Call to Action
"Label 500 samples → Train → Deploy dashboard"

---

## 📞 Contact & Repository

- **GitHub**: https://github.com/Fahri-Hilm/Analisis-SVM-Timnas
- **Docs**: README.md, QUICKSTART_LAYERED.md
- **Demo**: notebooks/layered_analytics_demo.ipynb

---

**Status: ✅ Infrastructure Complete | 🟡 Awaiting Labeled Data | 🔵 Ready for Training**
