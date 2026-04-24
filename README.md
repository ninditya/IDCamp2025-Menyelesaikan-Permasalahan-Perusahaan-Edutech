# Prediksi Dropout Mahasiswa: Jaya Jaya Institut

![Python](https://img.shields.io/badge/Python-3.10.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Ready_to_Deploy-yellow)
![License](https://img.shields.io/badge/MIT-green)

> Proyek ini membangun sistem deteksi dini mahasiswa berisiko dropout di Jaya Jaya Institut menggunakan data historis tahun 2008 sampai 2015.

---

## Daftar Isi

- [Business Understanding](#business-understanding)
- [Permasalahan Bisnis](#permasalahan-bisnis)
- [Cakupan Proyek](#cakupan-proyek)
- [Persiapan](#persiapan)
- [Struktur Proyek](#struktur-proyek)
- [Business Dashboard](#business-dashboard)
- [Menjalankan Sistem Machine Learning](#menjalankan-sistem-machine-learning)
- [Kesimpulan](#kesimpulan)
- [Rekomendasi Action Items](#rekomendasi-action-items)

---

## Business Understanding

Jaya Jaya Institut adalah perguruan tinggi yang sudah berdiri sejak tahun 2000. Selama lebih dari dua dekade, institusi ini sudah menghasilkan banyak lulusan. Tapi di balik itu, ada satu masalah yang terus jadi perhatian: cukup banyak mahasiswa yang tidak menyelesaikan studinya alias dropout.

Dropout yang tinggi bisa berdampak pada beberapa hal:
- **Reputasi** - kepercayaan calon mahasiswa dan mitra industri bisa menurun.
- **Finansial** - biaya operasional tidak sebanding kalau banyak mahasiswa tidak sampai lulus.
- **Sosial** - mahasiswa yang dropout kehilangan peluang karir yang seharusnya bisa mereka raih.
- **Efisiensi** - beasiswa, fasilitas, dan tenaga pengajar jadi tidak dimanfaatkan secara optimal.

---

## Permasalahan Bisnis

Pertanyaan bisnis yang ingin dijawab (sesuai prinsip SMART):

1. **Berapa besar tingkat dropout** mahasiswa Jaya Jaya Institut **pada periode 2008 sampai 2015**, dan **bagaimana distribusinya** di tiap jurusan?
2. **Faktor demografis dan akademik apa** yang paling berpengaruh terhadap dropout **berdasarkan data 2008 sampai 2015**, supaya bisa dijadikan indikator peringatan dini?
3. **Pada semester keberapa** (1 atau 2) dropout paling sering terjadi, dan **metrik akademik apa** yang paling prediktif pada periode tersebut?
4. **Bagaimana dampak status finansial** (tuition up-to-date, debtor, scholarship) terhadap dropout rate **dalam data historis** tersebut?

---

## Cakupan Proyek

1. EDA komprehensif (univariate, bivariate, multivariate, statistical testing)
2. Business Dashboard interaktif di Looker Studio
3. Model ML (Random Forest / XGBoost dipilih otomatis via GridSearchCV) dengan perbandingan 4 algoritma dan hyperparameter tuning
4. Prototype Streamlit yang siap di-deploy ke Community Cloud

**Metrik Utama:** F1-Score dan Recall untuk kelas Dropout.

---

## Persiapan

### Sumber Data

Dataset yang digunakan pada proyek ini bisa diakses di:

- **Repository Dicoding:** [students_performance/data.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)
- **URL Langsung:** `https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv`

Dataset berisi **4.424 baris** dan **37 kolom** informasi demografis, akademik, dan sosio-ekonomi mahasiswa beserta status kelulusan mereka.

### Versi Python

**Python 3.10.12**

### Setup Environment

**1. Ekstrak Berkas ZIP**

Ekstrak file `ninditya_sna-submission.zip`, lalu masuk ke folder proyek:

```bash
cd submission-idcamp-expert-2
```

**2. Buat dan Aktifkan Virtual Environment**

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Jalankan Notebook** (untuk melatih dan menyimpan model)

```bash
jupyter notebook notebook.ipynb
```

Jalankan semua cell dari atas ke bawah. Model akan tersimpan otomatis di folder `model/`.

**5. Jalankan Aplikasi Streamlit**

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`.

---

## Struktur Proyek

```
submission-idcamp-expert-2/
├── model/
│   ├── pipeline.pkl            # Model pipeline (preprocessor + best classifier)
│   └── feature_names.pkl       # Daftar nama fitur yang digunakan model
├── ninditya_sna-dashboard/     # Screenshot 4 halaman dashboard Looker Studio
│   ├── ninditya_sna-dashboard.png      # Halaman 1 - Overview
│   ├── ninditya_sna-dashboard-2.png    # Halaman 2 - Faktor Akademik
│   ├── ninditya_sna-dashboard-3.png    # Halaman 3 - Finansial & Demografi
│   ├── ninditya_sna-dashboard-4.png    # Halaman 4 - Rekomendasi & Intervensi
│   ├── LOOKER_HALAMAN4_BLUEPRINT.md    # Blueprint Halaman 4 dashboard
│   └── TRANSKRIP_VIDEO_DASHBOARD.md    # Transkrip narasi video 5 menit
├── ninditya_sna-dicoding-video/
│   └── ninditya_sna-dicoding-video.mp4 # Video demo aplikasi (~5 menit)
├── data/
│   ├── data_dashboard.csv      # Data bersih untuk Looker Studio (dihasilkan oleh notebook)
│   ├── action_items_data.csv   # Data statis untuk Halaman 4 Looker Studio (4 prioritas)
│   └── funnel_data.csv         # Data funnel 4.424 → 1.421 → 1.325 untuk chart Halaman 4
├── notebook.ipynb              # Notebook EDA + Modeling (semua cell sudah dijalankan)
├── app.py                      # Aplikasi Streamlit (PEP 8 + logging + docstring)
├── utils.py                    # Helper functions (EDA, evaluasi, I/O artifact)
├── README.md                   # Dokumentasi proyek (file ini)
└── requirements.txt            # Dependencies dengan versi ter-pin
```

---

## Business Dashboard

Dashboard interaktif dibuat menggunakan **Looker Studio** dengan **4 halaman**, masing-masing menggali satu dimensi masalah dropout.

**Link Dashboard:** https://datastudio.google.com/reporting/580a8000-c31f-4026-82a7-af1114ba5712/page/pH6vF

| Halaman | Fokus | Insight Utama |
|---------|-------|---------------|
| **1. Overview** | Distribusi keseluruhan | 4 KPI, dropout per jurusan & gender, filter interaktif |
| **2. Faktor Akademik** | Performa semester 1 & 2 | Mahasiswa 0 MK lulus Sem 1 → dropout 79%; scatter admission grade |
| **3. Finansial & Demografi** | UKT, beasiswa, usia | UKT menunggak & debitur → dropout sangat tinggi; usia > 25 thn berisiko |
| **4. Rekomendasi & Intervensi** | Estimasi dampak action items | Before/after dropout rate per segmen; ~1.325 mahasiswa bisa diselamatkan |

Fitur dashboard:
- 4 KPI Cards: Total Mahasiswa (4.424), Jumlah Dropout (1.421), Dropout Rate (32,12%), Graduate Rate (49,93%)
- Filter interaktif: Course, Gender, Age Range, Scholarship — berlaku untuk semua chart
- Halaman 4 menggunakan data source `data/action_items_data.csv` (Google Sheets statis)

Screenshot dashboard tersedia di folder `ninditya_sna-dashboard/`.

---

## Menjalankan Sistem Machine Learning

**Akses Online (Streamlit Community Cloud):**

*(akan diperbarui setelah di-deploy ke Streamlit Community Cloud)*

**Atau jalankan secara lokal:**

```bash
streamlit run app.py
```

**Cara menggunakan:**

1. Buka tab **🤖 Solusi Machine Learning** untuk melihat penjelasan alur pemodelan, perbandingan algoritma, metrik evaluasi, dan feature importance.
2. Buka tab **🔍 Prediksi Risiko Dropout** — isi data akademik (MK lulus, evaluasi, nilai), status finansial (UKT, tunggakan, beasiswa), data pribadi (gender, program studi), dan latar belakang keluarga.
3. Klik tombol **Prediksi Risiko Dropout**.
4. Lihat hasilnya: probabilitas dropout, tingkat risiko, gauge chart, dan rekomendasi tindakan.

---

## Kesimpulan

Berdasarkan hasil analisis data historis tahun 2008 sampai 2015, **tingkat dropout mahasiswa Jaya Jaya Institut mencapai sekitar 32%**. Angka ini cukup besar dan perlu segera ditangani.

### Ranking Feature Importance Global (Top 3)

| Rank | Fitur | Importance | Keterangan |
|------|-------|-----------|------------|
| #1 | `Curricular_units_2nd_sem_approved` | 28,6% | MK lulus semester 2 — sinyal terkuat |
| #2 | `Tuition_fees_up_to_date` | 10,7% | Status UKT — terpenting dari faktor finansial |
| #3 | `Curricular_units_1st_sem_enrolled` | 4,9% | MK diambil semester 1 |

### Faktor Akademik

1. **`Curricular_units_2nd_sem_approved`** (importance 28,6%, rank #1 global) — fitur terpenting. Mahasiswa yang sedikit lulus unit di semester 2 punya risiko dropout sangat tinggi.
2. **`Curricular_units_1st_sem_enrolled`** (importance 4,9%, rank #3 global) — jumlah MK yang diambil di semester 1; mahasiswa yang langsung mengambil banyak MK namun tidak mampu menyelesaikannya sangat berisiko.
3. **`Curricular_units_1st_sem_approved`** (importance 3,3%) — indikator awal yang sangat prediktif: mahasiswa dengan 0–4 MK lulus di semester 1 memiliki dropout rate 74–88%.

### Faktor Finansial

1. **Status pembayaran UKT (`Tuition_fees_up_to_date`)** (importance 10,7%, rank #2 global) — fitur terpenting kedua secara keseluruhan. Mahasiswa dengan UKT menunggak memiliki dropout rate **94,0%** (486 mhs), jauh di atas rata-rata 39,1%.
2. **Status debitur (`Debtor`)** (importance 3,0%) — mahasiswa berstatus debitur memiliki dropout rate **75,5%** (413 mhs).
3. **Penerima beasiswa (`Scholarship_holder`)** (importance 2,7%) — penerima beasiswa hanya dropout **13,8%** (969 mhs), vs non-penerima **48,4%**.

### Faktor Demografi

1. **Usia saat mendaftar** - mahasiswa yang masuk di usia lebih tua (di atas 25 tahun) lebih sering dropout, kemungkinan karena harus membagi waktu dengan pekerjaan atau tanggung jawab keluarga.
2. **Jurusan** - beberapa jurusan punya dropout rate jauh di atas rata-rata; perlu ada audit kurikulum.
3. **Status pernikahan** - mahasiswa dengan status menikah atau bercerai menunjukkan risiko yang lebih tinggi.

### Performa Model

Model terbaik dipilih secara otomatis dari GridSearchCV (perbandingan **Random Forest** vs **XGBoost**) berdasarkan F1-Score tertinggi:

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | 92,42% |
| **F1-Score (Dropout)** | 90,37% |
| **ROC-AUC** | 97,27% |

Model ini siap dipakai sebagai sistem deteksi dini dropout di Jaya Jaya Institut.

---

## Rekomendasi Action Items

### 1. Program Pembayaran Cicilan Darurat (Prioritas TINGGI)

Mahasiswa yang tidak membayar UKT tepat waktu punya dropout rate yang sangat tinggi. **Rekomendasi:** Terapkan skema cicilan 0% bunga dengan periode 6 bulan, ditambah diskon 10% biaya semester berikutnya kalau cicilan lunas tepat waktu. **Target:** Turunkan dropout pada segmen ini lebih dari 50% dalam 2 semester.

### 2. Early Warning System Semester 1 (Prioritas TINGGI)

Mahasiswa yang sedikit sekali berhasil lulus unit di semester 1 punya probabilitas dropout yang sangat tinggi. **Rekomendasi:** Buat trigger alert otomatis ke akademik advisor setiap akhir semester 1 untuk mahasiswa di segmen ini, diikuti wajib konseling akademik 4 sesi ditambah program mentoring peer-to-peer selama 1 semester. **Target:** Pertahankan lebih dari 30% mahasiswa berisiko sampai semester 2.

### 3. Ekspansi Beasiswa Berbasis Kinerja (Prioritas SEDANG)

Penerima beasiswa punya dropout rate jauh lebih rendah dibanding yang tidak menerima beasiswa. **Rekomendasi:** Tambah kuota beasiswa 15%, khususnya untuk mahasiswa usia 20 sampai 23 tahun dengan nilai masuk di atas atau sama dengan 120, karena segmen ini masih menunjukkan angka dropout yang bisa ditekan dengan dukungan finansial.

### 4. Program Khusus Mahasiswa Dewasa di Atas 25 Tahun (Prioritas SEDANG)

Mahasiswa yang masuk di usia lebih dari 25 tahun punya dropout rate jauh di atas rata-rata. **Rekomendasi:** Buat komunitas khusus mahasiswa dewasa, sediakan kelas evening-only untuk jurusan populer, dan berikan fleksibilitas deadline tugas. **Target:** Turunkan dropout segmen ini menjadi kurang dari 35% dalam 1 tahun.

### 5. Audit Kurikulum Jurusan dengan Dropout Tinggi (Prioritas JANGKA PANJANG)

Beberapa jurusan punya dropout rate sangat tinggi meski jumlah mahasiswanya tidak banyak. **Rekomendasi:** Lakukan audit kurikulum dalam 3 bulan, evaluasi beban SKS, dan pertimbangkan merger dengan jurusan serupa kalau ROI tidak tercapai dalam 2 tahun akademik.

### 6. Integrasi Model ke Sistem Informasi Akademik

Integrasikan model prediksi ke sistem informasi akademik yang sudah ada. Mahasiswa dengan probabilitas dropout lebih dari 60% otomatis masuk daftar pantauan, dan dosen pembimbing langsung dapat notifikasi setiap akhir semester. Model perlu di-retrain minimal sekali setahun dengan data terbaru.

---

## Author

**Ninditya**  
Email: ninditya.sna025@gmail.com  
Dicoding Username: ninditya_sna
