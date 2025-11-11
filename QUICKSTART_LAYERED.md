# 🚀 Quick Start Guide - Layered Analytics

## TL;DR - Cara Cepat Mulai

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Label some data (minimal 300-500 per class)
# - Ikuti panduan di data/labelled/LABELING_GUIDE.md
# - Simpan sebagai data/labelled/labelled_comments.csv

# 3. Jalankan demo notebook
jupyter notebook notebooks/layered_analytics_demo.ipynb

# 4. Train semua layer dan lihat hasilnya!
```

## 📋 Workflow Step-by-Step

### Step 1: Persiapan Environment
```bash
cd analisis-sentimen-timnas
pip install -r requirements.txt
```

### Step 2: Persiapan Data Berlabel

1. **Export subset komentar** dari data yang sudah ada:
```python
import pandas as pd

# Load existing data
df = pd.read_csv('data/processed/final_processed_dataset_real.csv')

# Sample 500 random comments
sample = df.sample(n=500, random_state=42)

# Export untuk labeling
sample[['comment_id', 'comment_text']].to_csv(
    'data/labelled/to_label.csv', 
    index=False
)
```

2. **Label manual** menggunakan template:
   - Buka `data/labelled/LABELING_GUIDE.md`
   - Ikuti definisi label
   - Isi kolom: sentiment, emotions, aspects, toxicity, stance, intent
   - Simpan sebagai `labelled_comments.csv`

3. **Format CSV yang diharapkan**:
```csv
comment_id,text,sentiment,emotions,aspects,toxicity,stance,intent
1,"Pelatih harus diganti!",negatif,"marah,kecewa","pelatih,strategi",non-toxic,kontra,komplain
2,"Kapan bisa lolos?",netral,sedih,,non-toxic,tidak_jelas,pertanyaan
```

### Step 3: Training Models

Open `notebooks/layered_analytics_demo.ipynb` dan run semua cell secara berurutan.

Atau gunakan script Python:

```python
import sys
sys.path.append('src')

from layered_classifier import EmotionClassifier
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Load labeled data
import pandas as pd
df = pd.read_csv('data/labelled/labelled_comments.csv')

texts = df['text'].tolist()
emotion_labels = df['emotions'].apply(
    lambda x: x.split(',') if x else []
).tolist()

# Train emotion classifier
emotion_clf = EmotionClassifier(
    emotion_labels=config['layered_analytics']['emotion_labels']
)
emotion_clf.train(texts, emotion_labels)

# Evaluate
metrics = emotion_clf.evaluate(texts, emotion_labels)
print(f"Emotion F1: {metrics['macro_f1']:.4f}")

# Save
emotion_clf.save(
    'models/emotion_svm.pkl',
    'models/emotion_vectorizer.pkl',
    'models/emotion_mlb.pkl'
)
```

### Step 4: Inference pada Data Baru

```python
# Load trained model
emotion_clf = EmotionClassifier(
    emotion_labels=config['layered_analytics']['emotion_labels']
)
emotion_clf.load(
    'models/emotion_svm.pkl',
    'models/emotion_vectorizer.pkl',
    'models/emotion_mlb.pkl'
)

# Predict
new_texts = [
    "Kecewa berat dengan hasil ini",
    "Mantap, terus semangat!"
]

predictions, confidences = emotion_clf.predict(new_texts)
print(predictions)
# Output: [['kecewa', 'sedih'], ['senang', 'bangga']]
```

### Step 5: Export Hasil Lengkap

```python
from src.utils.layered_utils import create_layered_output_dataframe

# Assume you have predictions from all layers
output_df = create_layered_output_dataframe(
    texts=new_texts,
    sentiment_labels=sentiment_pred,
    sentiment_confidence=sentiment_conf,
    emotion_labels=emotion_pred,
    aspect_labels=aspect_pred,
    toxicity_labels=toxicity_pred,
    toxicity_scores=toxicity_scores,
    stance_labels=stance_pred,
    stance_confidence=stance_conf,
    intent_labels=intent_pred,
    intent_confidence=intent_conf
)

# Save
output_df.to_csv('data/processed/layered_predictions.csv', index=False)
```

## 🎯 Target Jumlah Data per Layer

Untuk hasil optimal:

| Layer | Type | Min Samples | Recommended |
|-------|------|-------------|-------------|
| Emotion | Multi-label | 200/label | 300-500/label |
| Aspect | Multi-label | 200/label | 300-500/label |
| Toxicity | Binary | 300 total | 500-1000 total |
| Stance | 3-class | 200/class | 300-400/class |
| Intent | 6-class | 150/class | 200-300/class |

**Note**: Multi-label bisa overlap, jadi total unique instances bisa lebih sedikit.

## ⚙️ Konfigurasi

Edit `config.yaml` untuk customize:

```yaml
layered_analytics:
  enable_emotion: true      # Toggle on/off
  enable_aspect: true
  enable_toxicity: true
  enable_stance: true
  enable_intent: true
  
  low_confidence_threshold: 0.60  # Flag prediksi ragu
  toxicity_threshold: 0.50        # Threshold toxicity score
  
  emotion_labels:           # Customize labels
    - "marah"
    - "kecewa"
    # ... add more
```

## 📊 Output yang Dihasilkan

### 1. Models (`.pkl` files)
- `models/emotion_svm.pkl` + vectorizer + mlb
- `models/aspect_svm.pkl` + vectorizer + mlb
- `models/toxicity_svm.pkl` + vectorizer + encoder
- `models/stance_svm.pkl` + vectorizer + encoder
- `models/intent_svm.pkl` + vectorizer + encoder

### 2. Metrics (`.json` files)
- `models/metrics_emotion.json`
- `models/metrics_aspect.json`
- `models/metrics_toxicity.json`
- `models/metrics_stance.json`
- `models/metrics_intent.json`

### 3. Predictions (`.csv` files)
- `data/processed/layered_predictions.csv`

Contoh output:
```csv
text,sentiment,sentiment_confidence,emotions,aspects,toxicity_label,toxicity_score,stance,intent,low_confidence_flag
"Pelatih harus diganti!",negatif,0.89,"marah,kecewa","pelatih,strategi",non-toxic,0.12,kontra,komplain,False
```

### 4. Visualizations
- `results/visualizations/layered_distributions.png`
- Distribution charts per layer

### 5. Reports
- `results/reports/layered_analytics_report.txt`
- Comprehensive metrics summary

## 🐛 Troubleshooting

### Error: "File not found: labelled_comments.csv"
**Solution**: Buat file ini dulu! Ikuti Step 2 di atas.

### Error: "Not enough samples for training"
**Solution**: Label lebih banyak data. Minimal 100 per class untuk mulai (200+ recommended).

### Warning: "Low F1-score"
**Causes**:
1. Data tidak cukup
2. Data tidak seimbang → gunakan `class_weight='balanced'`
3. Label tidak konsisten → review labeling guide
4. Text preprocessing kurang bagus → enhance normalisasi

**Solutions**:
- Tambah data berlabel
- Check inter-annotator agreement
- Tune hyperparameters (C, gamma)
- Experiment dengan feature engineering

### Error: "Import error: module not found"
**Solution**: 
```bash
pip install -r requirements.txt
# or specific:
pip install scikit-learn pandas numpy matplotlib seaborn pyyaml joblib
```

## 💡 Tips & Best Practices

### 1. Labeling Quality
- **Consistency is key**: Gunakan 2 annotator untuk 10% sample
- **Clear guidelines**: Dokumentasikan edge cases
- **Regular review**: Check inter-annotator agreement (target >0.7)

### 2. Data Balance
- Aim for balanced classes (ratio < 3:1)
- Use `class_weight='balanced'` if imbalanced
- Consider oversampling minority class (SMOTE)

### 3. Iterative Improvement
- Start small (100-200 per class)
- Train → Evaluate → Identify errors → Label more → Retrain
- Focus labeling effort on misclassified examples

### 4. Threshold Tuning
- Default 0.5 might not be optimal
- Use precision-recall curves
- Adjust per use case (strict vs lenient)

### 5. Production Deployment
- Validate on held-out test set
- Monitor drift (data distribution changes)
- Retrain periodically with new data
- Log predictions + confidence for analysis

## 📚 Resources

- **Labeling Guide**: `data/labelled/LABELING_GUIDE.md`
- **Demo Notebook**: `notebooks/layered_analytics_demo.ipynb`
- **Full README**: `README.md`
- **Config**: `config.yaml`

## 🆘 Need Help?

1. Check documentation di `README.md`
2. Lihat example di notebook demo
3. Review error messages carefully
4. Open issue di GitHub repo

---

**Happy Analyzing! 🚀**
