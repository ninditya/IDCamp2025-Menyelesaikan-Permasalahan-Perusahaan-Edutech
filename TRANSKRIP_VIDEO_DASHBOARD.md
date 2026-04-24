# Transkrip Video: Proyek Prediksi Dropout Mahasiswa Jaya Jaya Institut

**Durasi estimasi:** 5:00 menit  
**Alat:** Looker Studio (4 halaman) + Streamlit  
**Pembuat:** Ninditya (ninditya_sna)

---

## [00:00 - 00:15] Opening

Halo. Di video ini saya jelaskan proyek prediksi dropout mahasiswa Jaya Jaya Institut — dashboard Looker Studio empat halaman, aplikasi Streamlit, dan rekomendasi action items berbasis data.

---

## [00:15 - 02:00] Dashboard Looker Studio

**Halaman 1: Overview.**
Empat KPI: total mahasiswa 4.424, dropout 1.421, dropout rate 32,12%, graduate rate 49,93%. Ada empat filter interaktif — jurusan, gender, usia, beasiswa — yang berlaku untuk semua chart. Bar chart dropout per jurusan diurutkan dari tertinggi: Biofuel, Equinokultur, dan Teknik Informatika di posisi teratas.

**Halaman 2: Faktor Akademik.**
Temuan paling kritis: mahasiswa yang tidak lulus satu pun unit di semester 1 punya dropout rate 79%. Yang lulus 1–3 unit masih 62%. Yang lulus 4 unit ke atas turun drastis ke 19%. Pola yang sama terulang di semester 2. Tabel di bawahnya mengkonfirmasi: Graduate rata-rata lulus 6,23 unit di semester 1, sedangkan Dropout hanya 2,55 — ini menjadi landasan **Prioritas 1: sistem alert otomatis di akhir semester 1**.

**Halaman 3: Finansial & Demografi.**
Mahasiswa dengan UKT menunggak atau berstatus debitur punya dropout rate jauh di atas rata-rata. Sebaliknya, penerima beasiswa hanya dropout 13,8%. Mahasiswa di atas 25 tahun punya dropout rate tertinggi di kelompok usia — dasar **Prioritas 2 dan Prioritas 4**.

**Halaman 4: Rekomendasi & Estimasi Dampak.**
Halaman ini memvisualisasikan potensi intervensi secara langsung. Grouped bar chart membandingkan dropout rate saat ini versus target — P1 dari 81% turun ke 30%, P2 dari 82% turun ke 40%. Bar chart di kanan menunjukkan estimasi penyelamatan — P1 terbesar: 676 mahasiswa. Total empat segmen, **sekitar 1.325 mahasiswa bisa dipertahankan per tahun** jika intervensi berhasil.

---

## [02:00 - 03:45] Aplikasi Streamlit

**Tab pertama: Solusi Machine Learning.**
Menjelaskan alur pemodelan: EDA, SMOTE untuk mengatasi imbalance kelas, 5-fold cross-validation empat algoritma, lalu GridSearchCV tuning. XGBoost terpilih dengan Accuracy 92,42%, F1-Score 90,37%, dan ROC-AUC 97,27%.

Feature importance top 20: fitur terpenting adalah MK lulus semester 2 dengan importance 28,6%, status UKT 10,7%, MK diambil semester 1 sebesar 4,9% — mengkonfirmasi temuan di dashboard.

Di bawahnya ada visualisasi dampak intervensi dan lima rekomendasi dengan angka spesifik, langkah implementasi, dan target terukur per segmen.

**Tab kedua: Prediksi Risiko Dropout.**
Form 20 fitur terpenting. Saya demonstrasikan dengan nilai default: MK lulus semester 1 dan 2 masing-masing 5, UKT tepat waktu, tidak ada tunggakan, program studi Manajemen. Setelah klik Prediksi, aplikasi menampilkan probabilitas dropout 5,8%, gauge chart hijau, status Graduate, dan rekomendasi positif: pertahankan performa akademik, pertimbangkan program akselerasi, dan bisa berperan sebagai mentor bagi mahasiswa lain.

---

## [03:45 - 04:45] Penutup

Dashboard Looker Studio memperlihatkan **di mana masalahnya**, Streamlit menjelaskan **mengapa model memilih fitur-fitur itu**, dan rekomendasi action items memberikan **langkah konkret berbasis data** untuk setiap segmen berisiko.

Model ini siap diintegrasikan ke sistem informasi akademik untuk deteksi dini otomatis setiap akhir semester, tanpa perlu analisis manual.

Terima kasih sudah menonton.

---

## Catatan Produksi

**Estimasi kata per segmen:**
- Opening: ~35 kata (15 detik)
- Dashboard (4 halaman): ~230 kata (106 detik)
- Streamlit (2 tab): ~165 kata (76 detik)
- Penutup: ~60 kata (28 detik)
- **Total: ~490 kata narasi + ~30 detik jeda/transisi ≈ 5 menit**

**Tampilan layar yang disarankan per segmen:**
- Opening: Halaman 1 dashboard sudah terbuka
- Hal. 1: tunjuk KPI lalu geser ke bar chart jurusan
- Hal. 2: zoom ke bar chart Sem 1 (angka 79% terlihat jelas), lalu tunjuk tabel perbandingan
- Hal. 3: hover >25 Tahun, tunjuk chart UKT dan Beasiswa
- Hal. 4: tunjuk grouped bar chart (merah vs biru), lalu bar chart 676 P1, lalu angka 1.325
- Tab 1 Streamlit: scroll — metrik final → feature importance → chart dampak → expander P1–P5
- Tab 2: gunakan nilai default (Sem1=5, Sem2=5, UKT Ya, Tidak Debtor, Manajemen) → klik Prediksi → tunjuk gauge hijau 5,8%, status Graduate, rekomendasi positif
- Penutup: kembali ke halaman 4 dashboard, tunjuk angka 1.325

**Angka dashboard (basis 4.424 termasuk Enrolled) vs model (basis 3.630 tanpa Enrolled):**
- Sem1 0 unit: dashboard 79% | model 88,1%
- Usia >25 thn: dashboard ~55,89% | model 64,6%
- Gunakan angka dashboard saat menjelaskan Looker, angka model saat menjelaskan Streamlit
