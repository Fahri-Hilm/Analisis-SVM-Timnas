# Template untuk Pelabelan Manual - Layered Analytics

## Instruksi Umum
File ini berisi contoh komentar yang perlu dilabeli secara manual untuk melatih model multi-layer.

**PENTING:** 
- Setiap baris adalah satu komentar
- Untuk multi-label (emotion, aspect): pisahkan dengan koma (misal: marah,kecewa)
- Untuk single-label (sentiment, toxicity, stance, intent): pilih satu
- Kosongkan kolom jika tidak yakin atau tidak relevan

## Format CSV
comment_id,text,sentiment,emotions,aspects,toxicity,stance,intent

## Definisi Label

### 1. Sentiment (pilih 1)
- **positif**: pujian, kepuasan, dukungan eksplisit
- **netral**: informatif, observasi tanpa opini kuat
- **negatif**: kekecewaan, keluhan, kritik

### 2. Emotions (multi-label, pisahkan dengan koma)
- **marah**: ungkapan amarah/kemarahan
- **kecewa**: ungkapan kekecewaan
- **sedih**: ungkapan kesedihan
- **senang**: ungkapan kegembiraan
- **bangga**: ungkapan kebanggan
- **takut**: ungkapan ketakutan/kekhawatiran

Contoh: "marah,kecewa" jika komentar menunjukkan kedua emosi

### 3. Aspects (multi-label, pisahkan dengan koma)
- **manajemen**: terkait pengelolaan/organisasi tim
- **pelatih**: terkait pelatih/coach
- **pemain**: terkait pemain individu atau kolektif
- **strategi**: terkait taktik/strategi permainan
- **wasit**: terkait wasit/arbitrase
- **PSSI**: terkait federasi PSSI
- **federasi**: terkait FIFA/AFC atau federasi lain
- **fanbase**: terkait supporter/fan

Contoh: "pelatih,strategi" jika membahas keduanya

### 4. Toxicity (pilih 1)
- **toxic**: mengandung ujaran kasar, offensive, hate speech
- **non-toxic**: tidak mengandung ujaran bermasalah

### 5. Stance (pilih 1)
- **pro**: mendukung/setuju dengan isu/pihak tertentu
- **kontra**: menentang/tidak setuju
- **tidak_jelas**: netral atau ambigu

### 6. Intent (pilih 1)
- **pertanyaan**: bertanya informasi
- **komplain**: mengeluh/mengkritik
- **saran**: memberi saran/masukan
- **ajakan**: mengajak/memotivasi
- **humor**: bercanda/humor
- **informasi**: memberi informasi

## Contoh Pelabelan

```csv
comment_id,text,sentiment,emotions,aspects,toxicity,stance,intent
1,"Pelatih harus diganti, strateginya kacau!",negatif,"marah,kecewa","pelatih,strategi",non-toxic,kontra,komplain
2,"Kapan Indonesia bisa lolos Piala Dunia?",netral,sedih,,non-toxic,tidak_jelas,pertanyaan
3,"Mantap performanya, terus semangat!",positif,"senang,bangga",pemain,non-toxic,pro,ajakan
4,"PSSI bodoh! Manajemen sampah!",negatif,marah,"PSSI,manajemen",toxic,kontra,komplain
5,"Menurut saya sebaiknya latihan lebih intensif",netral,,strategi,non-toxic,tidak_jelas,saran
```

## Tips Pelabelan
1. Baca komentar dengan seksama, pahami konteks
2. Jangan terburu-buru; ambil waktu untuk setiap komentar
3. Jika ragu, konsultasikan dengan tim atau tandai untuk review
4. Untuk emotions dan aspects: bisa lebih dari satu, tapi pilih yang dominan saja
5. Toxicity: fokus pada kata/frasa kasar atau offensive
6. Stance: terkait posisi terhadap tim/manajemen/isu tertentu
7. Intent: fokus pada tujuan utama komentar

## Target Jumlah Data per Label (Rekomendasi)
- **Sentiment**: 300-500 per kelas (total ~1000-1500)
- **Emotion**: 200-300 per label (bisa overlap)
- **Aspect**: 200-300 per label (bisa overlap)
- **Toxicity**: 300-500 per kelas (fokus lebih pada toxic untuk deteksi)
- **Stance**: 200-400 per kelas
- **Intent**: 150-300 per kelas

## Alur Kerja
1. Export subset komentar dari `data/processed/` atau `data/raw/`
2. Labeli secara manual menggunakan template ini
3. Simpan sebagai CSV di folder `data/labelled/`
4. Nama file: `labelled_comments_[tanggal].csv`
5. Gunakan notebook training untuk train model

## Quality Control
- Minimal 2 annotator untuk 10% sampel (inter-annotator agreement)
- Diskusikan case yang controversial
- Dokumentasikan guideline khusus jika ada edge case

---

**File ini adalah template. Copy dan rename sesuai kebutuhan.**
**Lokasi penyimpanan: `data/labelled/labelled_comments_template.csv`**
