# 👕 VARStyle — Hybrid AI Outfit Recommendation System

VARStyle adalah sistem rekomendasi outfit berbasis Artificial Intelligence (AI) yang menggabungkan pendekatan **Rule-Based Recommendation** dan **Visual Similarity Recommendation** menggunakan pretrained Deep Learning model **MobileNetV2**.

Sistem ini membantu pengguna mendapatkan rekomendasi outfit yang relevan berdasarkan:

* body type,
* style preference,
* aktivitas,
* environment,
* dan preferensi warna.

Selain menghasilkan recommendation, sistem juga memberikan explainable recommendation agar pengguna memahami alasan rekomendasi yang diberikan.

---

# 🚀 Demo Deployment

🌐 Streamlit App:
https://varstyle-app-vita.streamlit.app/

📂 GitHub Repository:
https://github.com/vitatriutami/ai-outfit-recommendation-system

---

# 📌 Deskripsi Singkat Proyek

Di era fast fashion dan digitalisasi, pengguna sering mengalami kesulitan dalam menentukan outfit yang sesuai dengan:

* bentuk tubuh,
* konteks aktivitas,
* dan style personal.

Sebagian besar fashion recommendation system saat ini masih:

* berbasis kategori sederhana,
* belum mempertimbangkan visual consistency,
* dan belum memberikan explainable recommendation.

VARStyle dikembangkan untuk menjawab permasalahan tersebut melalui pendekatan:

## ✅ Rule-Based Recommendation

Filtering berdasarkan:

* body type,
* activity type,
* indoor/outdoor context,
* style compatibility,
* color preference.

## ✅ Visual Similarity Recommendation

Menggunakan:

* MobileNetV2 pretrained CNN,
* feature embedding,
* cosine similarity.

## ✅ Explainable Recommendation

Sistem memberikan alasan recommendation seperti:

* cocok untuk body type,
* sesuai style,
* sesuai aktivitas,
* memiliki visual similarity tinggi.

---

# 🧠 Teknologi yang Digunakan

* Python
* Streamlit
* TensorFlow / Keras
* MobileNetV2
* Scikit-learn
* NumPy
* Pandas
* Pillow

---

# 📂 Struktur Project

```bash
ai-outfit-recommendation-system/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── data/
│   ├── products.csv
│   └── fashion_embeddings.npy
│
├── images/
│
├── models/
│
└── PersonalOutfitRecommendation.ipynb
```

---

# 📊 Dataset

Dataset menggunakan:
Fashion Product Images Dataset (Kaggle)

🔗 Dataset:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Dataset berisi:

* gambar fashion,
* kategori pakaian,
* warna,
* usage,
* style metadata,
* dan atribut fashion lainnya.

---

# 🧠 Model AI

## MobileNetV2

Model pretrained MobileNetV2 digunakan sebagai:

* lightweight CNN feature extractor,
* penghasil visual embedding dari gambar fashion.

Embedding tersebut digunakan untuk:

* menghitung visual similarity,
* menjaga konsistensi estetika outfit recommendation.

---

# ⚙️ Petunjuk Setup Environment

## 1. Clone Repository

```bash
git clone https://github.com/vitatriutami/ai-outfit-recommendation-system.git
```

```bash
cd ai-outfit-recommendation-system
```

---

## 2. Buat Virtual Environment (Opsional)

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Tautan Model

Model menggunakan:

* MobileNetV2 pretrained ImageNet
* TensorFlow Keras Applications

Model otomatis akan diunduh oleh TensorFlow saat pertama kali dijalankan.

Dokumentasi:
https://keras.io/api/applications/mobilenet/

---

# ▶️ Cara Menjalankan Aplikasi

Jalankan aplikasi Streamlit menggunakan command berikut:

```bash
streamlit run app.py
```

Jika berhasil, aplikasi akan berjalan di browser melalui:

```bash
http://localhost:8501
```

---

# 🎯 Fitur Utama

## ✅ User Preference Input

Pengguna dapat memilih:

* body type,
* activity type,
* environment,
* style preference,
* preferred color.

## ✅ Hybrid Recommendation System

Menggabungkan:

* rule-based filtering,
* visual similarity recommendation.

## ✅ Explainable Recommendation

Menampilkan alasan recommendation secara transparan.

## ✅ Visual Similarity AI

Menggunakan cosine similarity antar embedding fashion item.

---

# 📈 Alur Sistem Recommendation

```text
User Preference
       ↓
Rule-Based Filtering
       ↓
MobileNetV2 Feature Extraction
       ↓
Visual Embedding
       ↓
Cosine Similarity
       ↓
Hybrid Scoring
       ↓
Final Recommendation
```

---

# 📌 Pengembangan Selanjutnya

Beberapa pengembangan yang direncanakan:

* upload image recommendation,
* outfit pairing recommendation,
* fine-tuning fashion embedding,
* collaborative filtering,
* personalization learning,
* mobile app integration.

---

# 👤 Author

Vita Tri Utami

* Dicoding ID: vitatriutami
* Email: [vitatriutami@gmail.com](mailto:vitatriutami@gmail.com)

---

# 📄 License

Project ini dikembangkan untuk kebutuhan Capstone Project dan pembelajaran AI Recommendation System.
