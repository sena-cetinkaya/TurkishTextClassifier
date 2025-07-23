# 🧠 Türkçe Cümle Duygu Sınıflandırma API'si (FastAPI + XGBoost)

Bu proje, kullanıcıdan alınan Türkçe cümleleri **olumlu**, **olumsuz** veya **nötr** olarak sınıflandıran bir makine öğrenimi API’sidir.  
Sınıflandırma işlemi, veri ön işleme (temizleme + lemmatizasyon) ve **XGBoost** algoritmasıyla gerçekleştirilir.  
Ayrıca, tahmin sonuçları şık bir HTML arayüzünde gösterilir.

---

## 🧠 Temel Özellikler

- 🧽 Cümle temizleme ve Türkçe kök bulma (lemmatization)
- 📊 TF-IDF vektörleştirme
- 🔤 Label encoding (duygu etiketlerini sayısal forma çevirme)
- 🤖 XGBoost algoritması ile sınıflandırma
- 🖼️ Bootstrap + CSS ile görsel arayüz (Jinja2)
- 🐳 Docker & Docker Compose desteği
- 🚀 FastAPI ile sunulan REST servisi

---

## 📁 Proje Yapısı

├── main.py # FastAPI app giriş noktası

├── routes.py # Tahmin endpoint'i ve HTML yönlendirme

├── model_trainer.py # Veri ön işleme ve model eğitimi

├── xgboost_model.pkl # Eğitilmiş model

├── tfidf_vectorizer.pkl # TF-IDF objesi

├── label_encoder.pkl # Label encoder

├── static/

│ └── styles.css # CSS tasarım dosyası

├── templates/

│ └── index.html # HTML arayüz

├── turkce_yorumlar_duygu_1000.csv # Eğitim verisi

├── Dockerfile # Uygulama Docker imajı

├── docker-compose.yml # PostgreSQL içermeyen, sadece API için yapılandırma

└── requirements.txt # Gerekli Python paketleri

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Gereksinimler

```pip install -r requirements.txt```

### 2️⃣ Model Eğitimi

```python model_trainer.py```

Bu işlem:

Türkçe yorumları işler (temizleme + lemmatizasyon)

Etiketleri sayıya çevirir

TF-IDF ile metni vektöre dönüştürür

XGBoost ile modeli eğitir

Model, vektörleştirici ve label encoder'ı .pkl olarak kaydeder

### 3️⃣ Uygulamayı Başlat

```uvicorn main:app --reload```

Arayüz: http://localhost:8000

Swagger UI: http://localhost:8000/docs

🧪 Tahmin Örneği

Anasayfada bir form aracılığıyla kullanıcıdan cümle alınır.
Sonuç aşağıdaki gibi gösterilir:

```Sınıflandırma Sonucu: olumlu / olumsuz / nötr```

Arka planda:

- Cümle temizlenir

- Vektöre çevrilir (TF-IDF)

- XGBoost modeliyle tahmin yapılır

- Sonuç tekrar etiket olarak gösterilir

🐳 Docker ile Çalıştırmak

```docker-compose up --build```

Uygulama 8000 portunda çalışır.

🧠 Kullanılan Teknolojiler

- Python

- FastAPI

- scikit-learn

- XGBoost

- TRNLP

- NLTK (stopwords)

- TF-IDF Vectorizer

- Jinja2 + Bootstrap

- Docker

📃 Lisans: Bu proje MIT lisansı ile açık kaynak olarak sunulmuştur.

👩‍💻 Geliştirici: Sena Çetinkaya

📧 [cetinkayasena96@gmail.com](cetinkayasena96@gmail.com)

🌐 GitHub: [github.com/kullaniciadi](https://github.com/sena-cetinkaya)



















