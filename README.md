# 👕 VARStyle — Hybrid AI Outfit Recommendation System

VARStyle adalah sistem rekomendasi outfit berbasis AI yang membantu pengguna menemukan item fashion yang sesuai dengan karakteristik tubuh, preferensi gaya, warna favorit, dan konteks aktivitas.

Proyek ini mengimplementasikan pendekatan **Hybrid Recommendation System** yang menggabungkan:

* Rule-Based Recommendation
* Content-Based Recommendation
* Visual Similarity menggunakan Computer Vision
* Explainable Recommendation

---

# 🚀 Live Demo

Streamlit App:

https://varstyle-app-vita.streamlit.app/

---

# 📌 Problem Statement

Banyak pengguna memiliki banyak pilihan pakaian namun tetap mengalami kesulitan menentukan outfit yang sesuai untuk digunakan.

Beberapa tantangan yang sering ditemui:

* Sulit memilih outfit sesuai aktivitas.
* Sulit menemukan item fashion yang cocok dengan karakteristik tubuh.
* Sulit menjaga konsistensi gaya berpakaian.
* Membutuhkan waktu cukup lama untuk menentukan outfit yang tepat.

VARStyle dikembangkan untuk membantu proses pengambilan keputusan tersebut melalui sistem rekomendasi fashion berbasis AI.

---

# 🎯 Project Objectives

Tujuan utama proyek ini adalah membangun sistem rekomendasi fashion yang mampu:

* Memahami preferensi pengguna.
* Mempertimbangkan body type pengguna.
* Mempertimbangkan konteks aktivitas.
* Memanfaatkan kemiripan visual antar produk fashion.
* Memberikan alasan mengapa rekomendasi diberikan.

---

# 🏗️ System Architecture

VARStyle menggunakan arsitektur Hybrid Recommendation System.

```text
User Preference
      │
      ▼
Rule-Based Filtering
      │
      ▼
Rule-Based Ranking
      │
      ▼
Visual Similarity Re-ranking
(MobileNetV2 + Cosine Similarity)
      │
      ▼
Hybrid Recommendation Score
      │
      ▼
Top-N Recommendations
      │
      ▼
Explanation Generator
```

---

# 🤖 AI Components

## Computer Vision Feature Extraction

Model yang digunakan:

* MobileNetV2 (Pretrained ImageNet)

Digunakan sebagai:

* Feature Extractor

Output:

* Embedding Vector

Embedding digunakan untuk menghitung tingkat kemiripan visual antar produk fashion.

---

## Similarity Measurement

Metode:

* Cosine Similarity

Digunakan untuk:

* Mengukur kemiripan visual antar item fashion.
* Mendukung proses visual recommendation.

---

# 📊 Recommendation Strategy

VARStyle menggabungkan beberapa layer rekomendasi:

## Layer 1 — Candidate Filtering

Filter berdasarkan:

* Body Type
* Formality
* Indoor / Outdoor Context

---

## Layer 2 — Rule-Based Scoring

Komponen skor:

| Feature           | Weight |
| ----------------- | ------ |
| Body Type Match   | 40%    |
| Style Match       | 25%    |
| Activity Context  | 20%    |
| Environment Match | 10%    |
| Color Preference  | 5%     |

---

## Layer 3 — Visual Similarity Re-ranking

Menggunakan:

* MobileNetV2 Embedding
* Cosine Similarity

---

## Layer 4 — Explainable Recommendation

Sistem menghasilkan alasan rekomendasi seperti:

* Suitable for your body type
* Matches activity context
* Consistent with your style
* Matches preferred color
* High visual similarity

---

# 📂 Dataset

Dataset yang digunakan:

Fashion Product Images Dataset

Source:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Dataset berisi:

* Product images
* Product category
* Product metadata
* Color information
* Usage information
* Gender information

---

# ⚙️ Metadata Enrichment

Karena dataset tidak menyediakan seluruh atribut yang dibutuhkan, dilakukan proses metadata enrichment:

## Added Features

* formal_level
* indoor_outdoor
* style_group
* body_type_match

Tujuan:

* Mendukung filtering.
* Mendukung scoring.
* Menambah konteks recommendation system.

---

# 📁 Project Structure

```text
ai-outfit-recommendation-system
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── data
│   ├── products.csv
│   └── fashion_embeddings.npy
│
├── images
│   └── fashion images
│
├── models
│   └── mobilenet_feature_extractor.keras
│
├── PersonalOutfitRecommendation.ipynb
│
└── README.md
```

---

# 🛠️ Installation

## Clone Repository

```bash
git clone https://github.com/vitatriutami/ai-outfit-recommendation-system.git

cd ai-outfit-recommendation-system
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / Mac

```bash
python -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

Jalankan Streamlit:

```bash
streamlit run app.py
```

Aplikasi akan tersedia di:

```text
http://localhost:8501
```

---

# 📖 Usage Guide

## Step 1

Pilih Body Type:

* Pear
* Apple
* Rectangle
* Hourglass

---

## Step 2

Pilih Activity Type:

* Casual
* Formal
* Sports
* Semi Formal

---

## Step 3

Pilih Environment:

* Indoor
* Outdoor

---

## Step 4

Pilih Fashion Style:

* Minimalist
* Sporty
* Streetwear
* Smart Casual
* Casual
* Ethnic

---

## Step 5

Pilih warna favorit.

---

## Step 6

Klik tombol:

```text
Generate AI Recommendation
```

---

## Step 7

Sistem akan menghasilkan:

* Top outfit recommendations
* Recommendation score
* Visual similarity score
* Explanation recommendation

---

# 📈 Current MVP Features

✅ Fashion dataset preparation

✅ Metadata enrichment

✅ MobileNetV2 feature extraction

✅ Visual similarity recommendation

✅ Hybrid recommendation engine

✅ Explainable recommendation

✅ Streamlit web application

✅ Deployment-ready assets

---

# ⚠️ Current Limitations

Beberapa keterbatasan pada MVP saat ini:

* Belum menggunakan user interaction data.
* Belum mengimplementasikan collaborative filtering.
* Bobot recommendation masih heuristic.
* Body type mapping masih rule-based.
* Belum mendukung multi-item outfit generation.
* Belum menggunakan fashion-specific embedding model.

---

# 🔮 Future Improvements

Roadmap pengembangan berikutnya:

* User image upload recommendation
* Outfit pairing recommendation
* Multi-item outfit generation
* Recommendation feedback loop
* Personalized recommendation history
* Fashion-specific embedding model
* Recommendation evaluation framework
* Collaborative filtering integration

---

# 📚 Technologies Used

Programming Language

* Python

Machine Learning

* TensorFlow
* Keras
* Scikit-Learn

Data Processing

* Pandas
* NumPy

Visualization

* Matplotlib

Web Application

* Streamlit

Deployment

* Streamlit Cloud

---

# 👩‍💻 Author

Vita Tri Utami

Dicoding ID: vitatriutami

Capstone Project — AI for Smart Recommendation Systems

---

# 📄 License

This project is developed for educational and capstone learning purposes.
