"""
Streamlit Application: Sistem Prediksi Risiko Dropout Mahasiswa Jaya Jaya Institut.

Tab 1 – Solusi Machine Learning: tahapan pemodelan dari notebook.
Tab 2 – Prediksi Dropout: form input 20 fitur terpenting + hasil prediksi.

Feature importance dipetakan menggunakan get_feature_names_out() (urutan output
ColumnTransformer), bukan feature_names_in_ (urutan input), agar mapping ke
feature_importances_ XGBoost benar.

Author: Ninditya
Email: ninditya.sna025@gmail.com
"""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================
MODEL_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "model" / "pipeline.pkl"
DROPOUT_THRESHOLD_HIGH = 0.75
DROPOUT_THRESHOLD_MED = 0.50

COURSE_MAP = {
    "Teknologi Produksi Biofuel": 33,
    "Desain Animasi & Multimedia": 171,
    "Layanan Sosial (Malam)": 8014,
    "Agronomi": 9003,
    "Desain Komunikasi": 9070,
    "Keperawatan Veteriner": 9085,
    "Teknik Informatika": 9119,
    "Equinokultur": 9130,
    "Manajemen": 9147,
    "Layanan Sosial": 9238,
    "Pariwisata": 9254,
    "Keperawatan": 9500,
    "Kebersihan Mulut": 9556,
    "Manajemen Periklanan & Pemasaran": 9670,
    "Jurnalisme & Komunikasi": 9773,
    "Pendidikan Dasar": 9853,
    "Manajemen (Malam)": 9991,
}

QUALIFICATION_MAP = {
    "Pendidikan Menengah / Kelas 12": 1,
    "Sarjana (S1)": 2,
    "Gelar Perguruan Tinggi": 3,
    "Master (S2)": 4,
    "Doktor (S3)": 5,
    "Kelas 9 - Tidak Selesai": 29,
    "Kelas 8": 30,
    "Pendidikan Dasar Siklus 3": 19,
    "Pendidikan Dasar Siklus 2": 38,
    "Tidak Bisa Baca/Tulis": 35,
    "Bisa Baca, Tanpa Kelas 4 SD": 36,
    "SD (Siklus 1)": 37,
    "Tidak Diketahui": 34,
}

OCCUPATION_MAP = {
    "Pelajar/Mahasiswa": 0,
    "Eksekutif / Direktur": 1,
    "Intelektual / Ilmiah": 2,
    "Teknisi Menengah": 3,
    "Staf Administrasi": 4,
    "Pekerja Jasa / Penjualan": 5,
    "Petani / Nelayan": 6,
    "Pekerja Industri / Konstruksi": 7,
    "Operator Mesin / Perakitan": 8,
    "Pekerja Tidak Terampil": 9,
    "Angkatan Bersenjata": 10,
    "Situasi Lain": 90,
}

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Jaya Jaya Institut - Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        logger.info("Model dimuat dari %s", MODEL_PATH)
        return joblib.load(MODEL_PATH)
    fallback = Path("model") / "pipeline.pkl"
    if fallback.exists():
        logger.info("Model dimuat dari fallback: %s", fallback)
        return joblib.load(fallback)
    logger.error("File model tidak ditemukan")
    return None


@st.cache_data
def get_feature_importances():
    """Kembalikan list (feature_name, importance) terurut descending.

    Menggunakan get_feature_names_out() bukan feature_names_in_ agar
    urutan nama fitur sesuai dengan urutan output ColumnTransformer
    yang dilihat oleh XGBoost saat training.
    """
    model = load_model()
    if model is None:
        return []
    preprocessor = model.named_steps["preprocessor"]
    clf = model.steps[-1][1]
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [
        n.replace("scale__", "").replace("pass__", "").replace("remainder__", "")
        for n in raw_names
    ]
    importances = clf.feature_importances_
    return sorted(zip(clean_names, importances), key=lambda x: -x[1])


# ============================================================================
# HELPERS
# ============================================================================
def build_input_dict(
    sem2_approved: int,
    tuition_up_to_date: str,
    sem1_enrolled: int,
    sem1_approved: int,
    debtor: str,
    sem1_no_eval: int,
    scholarship: str,
    sem2_credited: int,
    course: str,
    sem2_grade: float,
    sem2_evaluations: int,
    sem1_evaluations: int,
    sem2_enrolled: int,
    displaced: str,
    sem1_credited: int,
    application_order: int,
    gender: str,
    admission_grade: float,
    mothers_occ: str,
    fathers_qual: str,
) -> dict:
    """Bangun dict fitur model dari 20 input top feature + default median/modus untuk sisanya."""
    return {
        # --- Top 20 fitur terpenting (urutan sesuai feature importance) ---
        "Curricular_units_2nd_sem_approved": sem2_approved,
        "Tuition_fees_up_to_date": 1 if tuition_up_to_date == "Ya" else 0,
        "Curricular_units_1st_sem_enrolled": sem1_enrolled,
        "Curricular_units_1st_sem_approved": sem1_approved,
        "Debtor": 1 if debtor == "Ya" else 0,
        "Curricular_units_1st_sem_without_evaluations": sem1_no_eval,
        "Scholarship_holder": 1 if scholarship == "Ya" else 0,
        "Curricular_units_2nd_sem_credited": sem2_credited,
        "Course": COURSE_MAP[course],
        "Curricular_units_2nd_sem_grade": sem2_grade,
        "Curricular_units_2nd_sem_evaluations": sem2_evaluations,
        "Curricular_units_1st_sem_evaluations": sem1_evaluations,
        "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
        "Displaced": 1 if displaced == "Ya" else 0,
        "Curricular_units_1st_sem_credited": sem1_credited,
        "Application_order": application_order,
        "Gender": 1 if gender == "Laki-laki" else 0,
        "Admission_grade": admission_grade,
        "Mothers_occupation": OCCUPATION_MAP[mothers_occ],
        "Fathers_qualification": QUALIFICATION_MAP[fathers_qual],
        # --- Default median/modus dari dataset (fitur di luar top 20) ---
        "Age_at_enrollment": 20,
        "Application_mode": 1,
        "Curricular_units_1st_sem_grade": 12.0,
        "Curricular_units_2nd_sem_without_evaluations": 0,
        "Daytime_evening_attendance": 1,
        "Educational_special_needs": 0,
        "Fathers_occupation": 7,
        "GDP": 0.32,
        "Inflation_rate": 1.4,
        "International": 0,
        "Marital_status": 1,
        "Mothers_qualification": 1,
        "Nacionality": 1,
        "Previous_qualification": 1,
        "Previous_qualification_grade": 133.1,
        "Unemployment_rate": 11.1,
    }


def create_gauge_chart(probability: float) -> go.Figure:
    if probability > DROPOUT_THRESHOLD_HIGH:
        bar_color = "#e74c3c"
    elif probability > DROPOUT_THRESHOLD_MED:
        bar_color = "#f39c12"
    else:
        bar_color = "#27ae60"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        delta={"reference": 50, "valueformat": ".1f"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Probabilitas Dropout", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 33], "color": "#d5f5e3"},
                {"range": [33, 66], "color": "#fef9e7"},
                {"range": [66, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin={"t": 60, "b": 20, "l": 30, "r": 30},
        paper_bgcolor="white",
    )
    return fig


def get_risk_factors(
    sem1_approved: int, sem2_approved: int, sem2_grade: float,
    tuition_up_to_date: str, debtor: str,
) -> list:
    factors = []
    if tuition_up_to_date == "Tidak":
        factors.append("UKT tidak tepat waktu")
    if debtor == "Ya":
        factors.append("Memiliki tunggakan")
    if sem1_approved < 3:
        factors.append("Sedikit MK lulus Sem 1")
    if sem2_approved < 3:
        factors.append("Sedikit MK lulus Sem 2")
    if sem2_grade < 10:
        factors.append("Nilai rata-rata Sem 2 rendah")
    return factors


def render_recommendations(
    pred: int, sem1_approved: int, sem2_approved: int,
    sem2_grade: float, tuition_up_to_date: str, debtor: str,
) -> None:
    st.markdown("### 💡 Rekomendasi Tindakan")
    if pred == 1:
        recs = []
        if tuition_up_to_date == "Tidak" or debtor == "Ya":
            recs.append(
                "💰 Konsultasikan masalah **finansial** ke bagian keuangan — "
                "ajukan cicilan 0% bunga atau beasiswa darurat."
            )
        if sem1_approved < 3 or sem2_approved < 3:
            recs.append(
                "📚 Ikuti program **bimbingan akademik intensif** — "
                "jumlah mata kuliah yang lulus sangat rendah."
            )
        if sem2_grade < 10:
            recs.append(
                "🎯 Tingkatkan nilai dengan bergabung **study group** "
                "dan konsultasi rutin dengan dosen."
            )
        recs.append("👥 Jadwalkan **sesi konseling** dengan dosen wali.")
        recs.append("📅 Lakukan **monitoring aktif** setiap 2 minggu.")
        for rec in recs:
            st.markdown(f"- {rec}")
    else:
        st.markdown("- ✨ Pertahankan performa akademik saat ini.")
        st.markdown("- 📈 Pertimbangkan program akselerasi atau mata kuliah pilihan tambahan.")
        st.markdown("- 🤝 Bisa berperan sebagai mentor bagi mahasiswa lain.")


# ============================================================================
# PAGE RENDERERS
# ============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🎓 Jaya Jaya Institut")
        st.markdown("---")
        st.markdown("**Tentang Aplikasi**")
        st.info(
            "Sistem prediksi risiko dropout mahasiswa berbasis **XGBoost** "
            "yang dipilih otomatis via GridSearchCV. "
            "Form menggunakan **20 fitur terpenting** berdasarkan feature importance."
        )
        st.markdown("---")
        st.markdown("**Interpretasi Hasil**")
        st.success("✅ Probabilitas < 50%: **Risiko Rendah**")
        st.warning("⚠️ 50% – 75%: **Risiko Menengah**")
        st.error("🚨 > 75%: **Risiko Tinggi**")
        st.markdown("---")
        st.markdown("**Dikembangkan oleh:**")
        st.markdown("Ninditya | IDCamp 2025")


def render_ml_solution() -> None:
    """Tab 1 — Penjelasan solusi machine learning dari notebook."""
    st.markdown("## 🤖 Solusi Machine Learning")
    st.markdown(
        "Halaman ini menjelaskan alur pemodelan yang digunakan untuk membangun "
        "sistem prediksi dropout mahasiswa Jaya Jaya Institut."
    )

    # --- Business Problem ---
    st.markdown("---")
    st.markdown("### 🏫 Latar Belakang & Problem")
    st.markdown(
        """
Jaya Jaya Institut menghadapi masalah **angka dropout mahasiswa yang tinggi** sejak berdiri tahun 2000.
Dropout yang tidak terdeteksi lebih awal berdampak pada:

- **Reputasi institusi** — calon mahasiswa baru cenderung menghindari kampus dengan angka dropout tinggi
- **Pendapatan** — setiap mahasiswa yang keluar adalah kehilangan potensi pendapatan
- **Akreditasi** — tingkat kelulusan menjadi salah satu indikator mutu yang diukur
- **Mahasiswa itu sendiri** — dropout sering berujung pada kerugian finansial dan karir

**Solusi:** Membangun model klasifikasi biner yang memprediksi apakah seorang mahasiswa
akan **Dropout** atau **Graduate** berdasarkan data akademik, finansial, dan demografis.
        """
    )

    # --- Pipeline Overview ---
    st.markdown("---")
    st.markdown("### 🔄 Alur Pemodelan (End-to-End)")
    steps = [
        ("1️⃣", "Business Understanding",
         "Identifikasi masalah dropout, tentukan target variabel (Dropout vs Graduate), "
         "keluarkan data Enrolled yang statusnya belum final."),
        ("2️⃣", "Eksplorasi Data (EDA)",
         "Statistik deskriptif, cek missing values & duplikat, deteksi outlier via IQR, "
         "visualisasi distribusi, uji Chi-Square untuk fitur kategorikal, uji normalitas Shapiro-Wilk."),
        ("3️⃣", "Persiapan Data",
         "Encoding target (Dropout=1, Graduate=0), identifikasi fitur kontinu vs biner/ordinal, "
         "analisis korelasi, export data bersih ke CSV untuk dashboard Looker Studio."),
        ("4️⃣", "Pemodelan & Seleksi",
         "Train/test split 80:20 dengan stratifikasi. SMOTE untuk mengatasi imbalance kelas. "
         "5-fold Stratified CV membandingkan 4 algoritma. GridSearchCV tuning 2 kandidat terbaik."),
        ("5️⃣", "Evaluasi & Deployment",
         "Evaluasi final di test set yang belum pernah dilihat model. "
         "Pipeline terbaik disimpan ke file .pkl dan diintegrasikan ke aplikasi Streamlit ini."),
    ]
    for icon, title, desc in steps:
        with st.expander(f"{icon} **{title}**", expanded=True):
            st.markdown(desc)

    # --- Data Split ---
    st.markdown("---")
    st.markdown("### 📊 Dataset & Pembagian Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data (non-Enrolled)", "3.630 mahasiswa")
    col2.metric("Training Set (80%)", "2.904 mahasiswa")
    col3.metric("Test Set (20%)", "726 mahasiswa")
    col4.metric("Proporsi Dropout", "~39%")
    st.info(
        "Data Enrolled dikeluarkan karena statusnya belum final. "
        "Stratifikasi dipakai saat split agar proporsi kelas tetap seimbang di kedua set."
    )

    # --- Preprocessing ---
    st.markdown("---")
    st.markdown("### ⚙️ Preprocessing Pipeline")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ColumnTransformer**")
        st.markdown(
            "- **StandardScaler** → fitur kontinu (nilai, usia, tingkat pengangguran, dll.)\n"
            "- **Passthrough** → fitur biner & ordinal yang sudah berbentuk integer"
        )
    with col_b:
        st.markdown("**SMOTE (Oversampling)**")
        st.markdown(
            "- Sebelum SMOTE: Graduate=1.767, Dropout=1.137\n"
            "- Setelah SMOTE: Graduate=1.767, Dropout=1.767\n"
            "- Hanya diterapkan pada **data training** untuk mencegah data leakage"
        )

    # --- Model Comparison ---
    st.markdown("---")
    st.markdown("### 📈 Perbandingan 4 Model — 5-Fold Stratified Cross-Validation")
    st.caption("Metrik diukur pada data training yang sudah di-balance dengan SMOTE.")
    cv_data = {
        "Model": ["XGBoost", "Gradient Boosting", "Random Forest", "Logistic Regression"],
        "F1 Mean": [0.9137, 0.9114, 0.9093, 0.8861],
        "F1 Std (±)": [0.0078, 0.0136, 0.0109, 0.0158],
        "ROC-AUC": [0.9669, 0.9657, 0.9648, 0.9521],
    }
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)
    st.success("✅ **XGBoost** dan **Random Forest** dipilih sebagai kandidat untuk tahap tuning.")

    # --- GridSearchCV ---
    st.markdown("---")
    st.markdown("### 🔧 Tuning Hyperparameter — GridSearchCV")
    col_rf, col_xgb = st.columns(2)
    with col_rf:
        st.markdown("**Random Forest**")
        st.markdown("Grid: 36 kombinasi, 180 fits total")
        st.code(
            "n_estimators     : 200\n"
            "max_depth        : None\n"
            "min_samples_split: 5\n"
            "min_samples_leaf : 1\n"
            "Best CV F1       : 0.8689",
            language="text",
        )
    with col_xgb:
        st.markdown("**XGBoost** ✅ Terpilih")
        st.markdown("Grid: 54 kombinasi, 270 fits total")
        st.code(
            "n_estimators  : 300\n"
            "max_depth     : 6\n"
            "learning_rate : 0.05\n"
            "subsample     : 0.8\n"
            "Best CV F1    : 0.8712",
            language="text",
        )
    st.info(
        "Scoring GridSearchCV menggunakan **F1-Score** (bukan accuracy), karena "
        "mendeteksi mahasiswa dropout yang terlewat (false negative) lebih merugikan "
        "daripada false positive."
    )

    # --- Final Evaluation ---
    st.markdown("---")
    st.markdown("### 🏆 Evaluasi Model Final — Test Set")
    st.caption("XGBoost (Tuned) dievaluasi pada 726 data test yang belum pernah dilihat model.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "92.42%")
    m2.metric("F1-Score (Dropout)", "90.37%")
    m3.metric("ROC-AUC", "97.27%")
    report_data = {
        "Kelas": ["Graduate", "Dropout", "Macro Avg", "Weighted Avg"],
        "Precision": [0.94, 0.90, 0.92, 0.92],
        "Recall": [0.93, 0.91, 0.92, 0.92],
        "F1-Score": [0.94, 0.90, 0.92, 0.92],
        "Support": [442, 284, 726, 726],
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

    # --- Feature Importance ---
    st.markdown("---")
    st.markdown("### 📊 Feature Importance — Top 20 Fitur Paling Berpengaruh")
    st.caption(
        "Dipetakan menggunakan `get_feature_names_out()` agar urutan nama fitur sesuai "
        "dengan urutan output ColumnTransformer yang dilihat XGBoost saat training."
    )
    ranked = get_feature_importances()
    if ranked:
        top20 = ranked[:20]
        # Label: ganti prefix panjang agar tetap terbaca tanpa terpotong
        def _shorten(name: str) -> str:
            return (name
                    .replace("Curricular_units_1st_sem_", "Sem1 – ")
                    .replace("Curricular_units_2nd_sem_", "Sem2 – ")
                    .replace("_", " "))
        feat_names = [_shorten(r[0]) for r in top20]
        feat_imps = [round(r[1] * 100, 2) for r in top20]
        colors = [
            "#e74c3c" if i == 0 else "#e67e22" if i < 3 else
            "#3498db" if i < 7 else "#85c1e9"
            for i in range(len(feat_names))
        ]
        fig_imp = go.Figure(go.Bar(
            x=feat_imps[::-1],
            y=feat_names[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"  {v:.2f}%" for v in feat_imps[::-1]],
            textposition="outside",
            textfont={"size": 13, "color": "#2c3e50"},
        ))
        fig_imp.update_layout(
            height=650,
            margin={"l": 20, "r": 100, "t": 20, "b": 40},
            xaxis=dict(
                title="Feature Importance (%)",
                title_font={"size": 13},
                tickfont={"size": 12},
                range=[0, max(feat_imps) * 1.25],
            ),
            yaxis=dict(
                tickfont={"size": 13, "color": "#1a1a2e"},
                automargin=True,
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"color": "#1a1a2e", "family": "Arial, sans-serif"},
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Model belum dimuat.")

    # --- Key Findings ---
    st.markdown("---")
    st.markdown("### 🔍 Temuan Utama & Faktor Risiko Dropout")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        st.markdown("#### 📚 Performa Sem 1")
        st.metric("0 MK lulus → dropout", "88,1%", delta="647 mhs", delta_color="inverse")
        st.metric("1–4 MK lulus → dropout", "74,3%", delta="676 mhs", delta_color="inverse")
        st.metric("≥5 MK lulus → dropout", "15,1%", delta="2.307 mhs", delta_color="normal")
    with col_f2:
        st.markdown("#### 💰 Kondisi Finansial")
        st.metric("UKT Menunggak → dropout", "94,0%", delta="486 mhs", delta_color="inverse")
        st.metric("Debitur → dropout", "75,5%", delta="413 mhs", delta_color="inverse")
        st.metric("Penerima Beasiswa → dropout", "13,8%", delta="969 mhs", delta_color="normal")
    with col_f3:
        st.markdown("#### 🎓 Program Studi Berisiko")
        st.metric("Teknik Informatika", "86,8%", delta="106 mhs", delta_color="inverse")
        st.metric("Equinokultur", "65,0%", delta="120 mhs", delta_color="inverse")
        st.metric("Manajemen (Malam)", "63,6%", delta="214 mhs", delta_color="inverse")
    with col_f4:
        st.markdown("#### 👤 Demografi")
        st.metric("Usia > 25 tahun → dropout", "64,6%", delta="904 mhs", delta_color="inverse")
        st.metric("Laki-laki → dropout", "56,1%", delta="1.249 mhs", delta_color="inverse")
        st.metric("Kombinasi kritis (>25+sem1=0)", "99,6%", delta="265 mhs", delta_color="inverse")

    # --- Recommendations ---
    st.markdown("---")
    st.markdown("### 📋 Rekomendasi Action Items")
    st.caption(
        "Disusun berdasarkan **prioritas dampak** — segmen dengan dropout rate tertinggi "
        "mendapat intervensi paling awal. Rata-rata dropout institusi: **39,1%**."
    )

    # Impact visualization
    _labels = [
        "P1: Sem 1 < 5 MK Lulus",
        "P2: UKT / Tunggakan",
        "P3: Prodi Berisiko",
        "P4: Usia > 25 Thn",
    ]
    _curr_rates  = [81.0, 82.3, 67.2, 62.7]
    _tgt_rates   = [30.0, 40.0, 40.0, 45.0]
    _est_saved   = [676,  285,  158,  206]
    _n_students  = [1323, 673,  582,  1161]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mahasiswa Berisiko (4 Segmen)", "3.739", delta="per segmen, bisa tumpang tindih", delta_color="off")
    k2.metric("Estimasi Bisa Diselamatkan", "~1.325", delta="jika semua intervensi berhasil", delta_color="normal")
    k3.metric("Dampak Terbesar", "P1: ~676 mhs", delta="Sem 1 < 5 MK → alert dini", delta_color="normal")
    k4.metric("Rata-rata Dropout Segmen", "73,3%", delta="vs institusi 39,1%", delta_color="inverse")

    st.markdown("")
    _ch1, _ch2 = st.columns([3, 2])

    with _ch1:
        st.markdown("**Dropout Rate: Saat Ini vs Target Setelah Intervensi**")
        _fig_ba = go.Figure()
        _fig_ba.add_trace(go.Bar(
            name="Rate Saat Ini",
            y=_labels, x=_curr_rates, orientation="h",
            marker_color="#e74c3c",
            text=[f"{v:.0f}%" for v in _curr_rates],
            textposition="auto",
            textfont={"color": "white", "size": 13},
        ))
        _fig_ba.add_trace(go.Bar(
            name="Target Setelah Intervensi",
            y=_labels, x=_tgt_rates, orientation="h",
            marker_color="#27ae60",
            text=[f"{v:.0f}%" for v in _tgt_rates],
            textposition="auto",
            textfont={"color": "white", "size": 13},
        ))
        _fig_ba.update_layout(
            barmode="group",
            height=270,
            margin={"l": 10, "r": 30, "t": 10, "b": 30},
            xaxis=dict(title="Dropout Rate (%)", range=[0, 100], tickfont={"size": 12}),
            yaxis=dict(tickfont={"size": 12, "color": "#1a1a2e"}, automargin=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font={"size": 12}),
            paper_bgcolor="white", plot_bgcolor="white",
            font={"color": "#1a1a2e", "family": "Arial, sans-serif"},
        )
        st.plotly_chart(_fig_ba, use_container_width=True)

    with _ch2:
        st.markdown("**Estimasi Mahasiswa yang Bisa Diselamatkan**")
        _fig_sv = go.Figure(go.Bar(
            y=_labels, x=_est_saved, orientation="h",
            marker_color=["#e74c3c", "#e67e22", "#3498db", "#27ae60"],
            text=[f"{v} mhs" for v in _est_saved],
            textposition="outside",
            textfont={"size": 13, "color": "#1a1a2e"},
        ))
        _fig_sv.update_layout(
            height=270,
            margin={"l": 10, "r": 80, "t": 10, "b": 30},
            xaxis=dict(title="Mahasiswa", range=[0, max(_est_saved) * 1.45],
                       tickfont={"size": 12}),
            yaxis=dict(tickfont={"size": 12, "color": "#1a1a2e"}, automargin=True,
                       showticklabels=False),
            paper_bgcolor="white", plot_bgcolor="white",
            font={"color": "#1a1a2e", "family": "Arial, sans-serif"},
        )
        st.plotly_chart(_fig_sv, use_container_width=True)

    # Priority summary table
    _tbl_data = {
        "Prioritas": ["🔴 P1 – Intervensi Akademik Sem 1", "🔴 P2 – Intervensi Finansial",
                      "🟠 P3 – Prodi Berisiko Tinggi",    "🟡 P4 – Mahasiswa Dewasa > 25 Thn"],
        "Jumlah Mhs": _n_students,
        "Dropout Rate Kini": [f"{r:.1f}%" for r in _curr_rates],
        "Target Setelah Intervensi": [f"{t:.0f}%" for t in _tgt_rates],
        "Estimasi Diselamatkan": [f"~{s} mhs" for s in _est_saved],
    }
    st.dataframe(pd.DataFrame(_tbl_data), use_container_width=True, hide_index=True)
    st.caption("Segmen bisa tumpang tindih — satu mahasiswa bisa masuk beberapa segmen sekaligus.")

    st.markdown("---")
    st.markdown("**Detail per Prioritas:**")

    with st.expander("🔴 PRIORITAS 1 — Intervensi Akademik Semester Pertama (Dampak Terbesar)", expanded=True):
        st.markdown(
            """
**Dasar data:** `Curricular_units_2nd_sem_approved` adalah fitur terpenting (importance 28,6%).
Mahasiswa dengan 0–4 MK lulus di Sem 1 berjumlah **1.323 orang (36% dari total)**
dengan dropout rate 74–88%. Kombinasi usia > 25 tahun + 0 MK lulus mencapai dropout rate **99,6%** (265 orang).

**Action items:**
- **Sistem alert otomatis di akhir Sem 1:** Notifikasi langsung ke dosen wali untuk semua mahasiswa
  yang lulus < 5 MK, disertai skor probabilitas dropout dari model (target: integrasi ke SIAKAD
  sebelum Sem 2 dimulai)
- **Program "Rescue Semester":** Wajibkan sesi bimbingan akademik intensif minimum **3 kali per bulan**
  bagi mahasiswa dengan 0–4 MK lulus Sem 1, dimulai paling lambat minggu ke-2 Sem 2
- **Pengurangan beban SKS di Sem 2:** Rekomendasikan dosen wali untuk menurunkan jumlah MK dari
  rata-rata 6 menjadi **3–4 MK** bagi mahasiswa berisiko, untuk mencegah kegagalan berulang
            """
        )

    with st.expander("🔴 PRIORITAS 2 — Intervensi Finansial Dini (Dampak Sangat Tinggi)", expanded=True):
        st.markdown(
            """
**Dasar data:** `Tuition_fees_up_to_date` adalah fitur terpenting #2 (importance 10,7%).
486 mahasiswa dengan UKT menunggak memiliki dropout rate **94,0%**. Dari 226 mahasiswa yang
sekaligus menunggak dan berstatus debitur, dropout rate mencapai **95,1%**. Total 673 mahasiswa
dengan masalah finansial memiliki dropout rate **82,3%**. Penerima beasiswa hanya dropout **13,8%**
vs non-penerima **48,4%**.

**Action items:**
- **Deteksi finansial di awal semester:** Tandai mahasiswa yang belum melunasi UKT dalam
  **30 hari pertama** dan langsung tawarkan 3 opsi:
  (a) cicilan 0% bunga dibagi 3 bulan,
  (b) keringanan UKT 30–50% dengan verifikasi dokumen,
  (c) pendaftaran beasiswa darurat
- **Prioritas konversi ke beasiswa:** Target konversikan minimal **200 mahasiswa berisiko finansial
  ke skema beasiswa** per tahun — penerima beasiswa dropout rate 13,8% vs non-penerima 48,4%
- **SLA penanganan:** Setiap kasus tunggakan harus mendapat respons dari bagian keuangan dalam
  **7 hari kerja**, dengan eskalasi ke pimpinan jika belum selesai dalam 14 hari
            """
        )

    with st.expander("🟠 PRIORITAS 3 — Program Khusus per Program Studi Berisiko Tinggi"):
        st.markdown(
            """
**Dasar data:** `Course` adalah fitur terpenting #9 (importance 2,1%).
Teknik Informatika (86,8% dropout, 94% laki-laki, 106 mhs), Equinokultur (65,0%, 120 mhs),
Manajemen Malam (63,6%, 214 mhs), dan Pendidikan Dasar (59,9%, 142 mhs) secara konsisten
berada di atas rata-rata institusi 39,1%.

**Action items:**
- **Audit kurikulum dan beban studi** di 4 prodi berisiko tinggi — identifikasi MK dengan tingkat
  kegagalan tertinggi dan evaluasi kesesuaian materi dengan kemampuan rata-rata mahasiswa baru;
  target selesai sebelum tahun akademik berikutnya
- **Mentor sebaya (peer mentoring) per prodi:** Rekrut mahasiswa Sem 3–4 berprestasi sebagai
  mentor dengan rasio **1 mentor : 5 mentee** untuk mahasiswa Sem 1 berisiko di Teknik Informatika
  dan Equinokultur; berikan insentif berupa poin KRS atau beasiswa parsial
- **Evaluasi MK bottleneck:** Untuk MK dengan tingkat kegagalan > 50% di prodi berisiko,
  lakukan peer review mengajar dan sediakan Teaching Assistant tambahan di semester berikutnya
            """
        )

    with st.expander("🟡 PRIORITAS 4 — Program Fleksibilitas untuk Mahasiswa Dewasa & Laki-Laki"):
        st.markdown(
            """
**Dasar data:** `Gender` adalah fitur terpenting #17 (importance 1,7%).
Mahasiswa usia > 25 tahun (904 orang) dropout rate 64,6%; usia 23–25 tahun (257 orang) 56,0%.
Laki-laki (56,1% dropout) hampir 2× lebih berisiko dari perempuan (30,2%).

**Action items:**
- **Jalur kuliah fleksibel:** Bagi mahasiswa usia > 25 tahun, tawarkan opsi hybrid
  (sebagian online, sebagian tatap muka) dan ujian susulan tanpa penalti untuk yang bekerja penuh waktu
- **Penyesuaian beban studi tahun pertama:** Rekomendasikan maksimum **4 MK per semester**
  untuk mahasiswa baru usia > 25 tahun, dengan evaluasi setelah Sem 1 sebelum kembali ke beban penuh
- **Konseling karir-studi:** Sediakan sesi konseling khusus membantu mahasiswa dewasa
  menyeimbangkan kewajiban kerja/keluarga dengan studi — targetkan laki-laki usia 23–30 tahun
            """
        )

    with st.expander("🟢 PRIORITAS 5 — Monitoring Berkelanjutan & Peningkatan Model"):
        st.markdown(
            """
**Action items:**
- **Dashboard real-time:** Integrasikan model ke SIAKAD agar manajemen dapat memantau
  distribusi risiko dropout per prodi setiap bulan tanpa perlu menarik data manual
- **Retrain model tahunan:** Latih ulang model setiap awal tahun akademik menggunakan
  data 3 tahun terakhir, dengan target mempertahankan **F1-Score ≥ 90%** dan **AUC ≥ 95%**
- **Evaluasi dampak intervensi:** Ukur perubahan dropout rate di setiap segmen yang
  diintervensi setiap semester — jika dropout rate kelompok < 5 MK Sem 1 tidak turun
  > 10 poin persentase dalam 2 semester, eskalasi program ke pimpinan institusi
            """
        )


def render_input_form() -> dict:
    """Form input dengan 20 fitur terpenting berdasarkan feature importance yang benar."""
    st.markdown("### 📚 Data Akademik Semester 1 & 2")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Semester 1**")
        sem1_approved = st.number_input(
            "MK Lulus Sem 1 ⭐ #4", min_value=0, max_value=26, value=5, key="s1a",
            help="Fitur terpenting #4 (importance 3,3%). Jumlah mata kuliah lulus semester 1.",
        )
        sem1_enrolled = st.number_input(
            "Diambil Sem 1 ⭐ #3", min_value=0, max_value=26, value=6, key="s1e",
            help="Fitur terpenting #3 (importance 4,9%). Jumlah MK yang diambil semester 1.",
        )
        sem1_evaluations = st.number_input(
            "Total Evaluasi Sem 1 #12", min_value=0, max_value=45, value=6, key="s1ev",
            help="Jumlah total evaluasi/ujian di semester 1.",
        )
        sem1_credited = st.number_input(
            "Dikreditkan Sem 1 #15", min_value=0, max_value=20, value=0, key="s1c",
            help="Jumlah unit yang dikreditkan/diakui di semester 1.",
        )
        sem1_no_eval = st.number_input(
            "Tanpa Evaluasi Sem 1 #6", min_value=0, max_value=12, value=0, key="s1ne",
            help="Fitur terpenting #6 (importance 2,8%). Unit tanpa evaluasi semester 1.",
        )

    with col_s2:
        st.markdown("**Semester 2**")
        sem2_approved = st.number_input(
            "MK Lulus Sem 2 ⭐ #1", min_value=0, max_value=26, value=5, key="s2a",
            help="Fitur TERPENTING #1 (importance 28,6%). Jumlah mata kuliah lulus semester 2.",
        )
        sem2_enrolled = st.number_input(
            "Diambil Sem 2 #13", min_value=0, max_value=26, value=6, key="s2e",
            help="Jumlah MK yang diambil di semester 2.",
        )
        sem2_evaluations = st.number_input(
            "Total Evaluasi Sem 2 #11", min_value=0, max_value=45, value=6, key="s2ev",
            help="Jumlah total evaluasi/ujian di semester 2.",
        )
        sem2_grade = st.slider(
            "Nilai Rata-rata Sem 2 (0–20) #10", 0.0, 20.0, 12.0, step=0.5, key="s2g",
            help="Fitur terpenting #10 (importance 2,1%). Nilai rata-rata semester 2.",
        )
        sem2_credited = st.number_input(
            "Dikreditkan Sem 2 #8", min_value=0, max_value=20, value=0, key="s2c",
            help="Fitur terpenting #8 (importance 2,2%). Unit dikreditkan semester 2.",
        )

    st.markdown("---")
    st.markdown("### 💰 Status Finansial & Beasiswa")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        tuition_up_to_date = st.radio(
            "Pembayaran UKT Tepat Waktu? ⭐ #2", ["Ya", "Tidak"], horizontal=True,
            help="Fitur terpenting #2 (importance 10,7%). UKT menunggak → dropout 94,0%.",
        )
    with col_f2:
        debtor = st.radio(
            "Memiliki Tunggakan? ⭐ #5", ["Tidak", "Ya"], horizontal=True,
            help="Fitur terpenting #5 (importance 3,0%). Debitur → dropout 75,5%.",
        )
    with col_f3:
        scholarship = st.radio(
            "Pemegang Beasiswa? #7", ["Tidak", "Ya"], horizontal=True,
            help="Fitur terpenting #7 (importance 2,7%). Penerima beasiswa → dropout hanya 13,8%.",
        )

    st.markdown("---")
    st.markdown("### 👤 Data Pribadi & Pendaftaran")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        gender = st.radio(
            "Jenis Kelamin #17", ["Laki-laki", "Perempuan"], horizontal=True,
            help="Fitur terpenting #17 (importance 1,7%). Laki-laki → dropout 56,1%.",
        )
        displaced = st.radio(
            "Mahasiswa Displaced? #14", ["Tidak", "Ya"], horizontal=True,
            help="Fitur terpenting #14 (importance 1,9%).",
        )
    with col_p2:
        course = st.selectbox(
            "Program Studi #9",
            list(COURSE_MAP.keys()),
            index=list(COURSE_MAP.keys()).index("Manajemen"),
            help="Fitur terpenting #9 (importance 2,1%). Teknik Informatika → dropout 86,8%.",
        )
        application_order = st.slider(
            "Urutan Pilihan Prodi #16", 1, 9, 1,
            help="Fitur terpenting #16 (importance 1,7%). 1 = prioritas utama.",
        )
    with col_p3:
        admission_grade = st.slider(
            "Nilai Masuk (0–200) #18", 0.0, 200.0, 127.0, step=0.5,
            help="Fitur terpenting #18 (importance 1,6%). Median mahasiswa dropout: 123,6.",
        )

    st.markdown("---")
    st.markdown("### 👨‍👩‍👧 Latar Belakang Keluarga")
    col_k1, col_k2 = st.columns(2)

    with col_k1:
        mothers_occ = st.selectbox(
            "Pekerjaan Ibu #19",
            list(OCCUPATION_MAP.keys()),
            index=list(OCCUPATION_MAP.keys()).index("Pekerja Jasa / Penjualan"),
            help="Fitur terpenting #19 (importance 1,6%).",
        )
    with col_k2:
        fathers_qual = st.selectbox(
            "Kualifikasi Pendidikan Ayah #20",
            list(QUALIFICATION_MAP.keys()),
            index=0,
            help="Fitur terpenting #20 (importance 1,5%).",
        )

    return {
        "sem2_approved": sem2_approved,
        "tuition_up_to_date": tuition_up_to_date,
        "sem1_enrolled": sem1_enrolled,
        "sem1_approved": sem1_approved,
        "debtor": debtor,
        "sem1_no_eval": sem1_no_eval,
        "scholarship": scholarship,
        "sem2_credited": sem2_credited,
        "course": course,
        "sem2_grade": sem2_grade,
        "sem2_evaluations": sem2_evaluations,
        "sem1_evaluations": sem1_evaluations,
        "sem2_enrolled": sem2_enrolled,
        "displaced": displaced,
        "sem1_credited": sem1_credited,
        "application_order": application_order,
        "gender": gender,
        "admission_grade": admission_grade,
        "mothers_occ": mothers_occ,
        "fathers_qual": fathers_qual,
    }


def render_prediction_result(model, input_df: pd.DataFrame, inputs: dict) -> None:
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    dropout_prob = float(proba[1])

    logger.info("Prediksi — pred=%d, prob=%.4f", pred, dropout_prob)

    st.markdown("## 📊 Hasil Prediksi")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if pred == 1:
            if dropout_prob > DROPOUT_THRESHOLD_HIGH:
                st.error("## 🚨 RISIKO DROPOUT SANGAT TINGGI")
                risk_label = "SANGAT TINGGI"
            else:
                st.warning("## ⚠️ RISIKO DROPOUT MENENGAH-TINGGI")
                risk_label = "MENENGAH-TINGGI"
        else:
            st.success("## ✅ RISIKO DROPOUT RENDAH")
            risk_label = "RENDAH"

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Probabilitas Dropout", f"{dropout_prob * 100:.1f}%")
        mcol2.metric("Tingkat Risiko", risk_label)
        mcol3.metric("Prediksi Status", "Dropout" if pred == 1 else "Graduate")

        st.markdown("---")
        render_recommendations(
            pred,
            inputs["sem1_approved"], inputs["sem2_approved"],
            inputs["sem2_grade"], inputs["tuition_up_to_date"], inputs["debtor"],
        )

    with col_res2:
        st.plotly_chart(create_gauge_chart(dropout_prob), use_container_width=True)

        st.markdown("**Breakdown Probabilitas**")
        pb_col1, pb_col2 = st.columns(2)
        pb_col1.metric("Peluang Graduate", f"{proba[0] * 100:.1f}%")
        pb_col2.metric("Peluang Dropout", f"{dropout_prob * 100:.1f}%")

        if dropout_prob > DROPOUT_THRESHOLD_MED:
            factors = get_risk_factors(
                inputs["sem1_approved"], inputs["sem2_approved"],
                inputs["sem2_grade"], inputs["tuition_up_to_date"], inputs["debtor"],
            )
            if factors:
                st.markdown("**Faktor Risiko yang Teridentifikasi:**")
                for f in factors:
                    st.markdown(f"- ⚠️ {f}")


def render_prediction_tab() -> None:
    """Tab 2 — Form prediksi dropout."""
    st.markdown("### Sistem Prediksi Risiko Dropout Mahasiswa")
    st.markdown(
        "Isi data mahasiswa di bawah ini (**20 fitur terpenting** berdasarkan feature importance model), "
        "lalu klik **Prediksi** untuk mengetahui risiko dropout."
    )
    st.info(
        "**16 fitur lainnya** menggunakan nilai median/modus dari dataset "
        "(Age=20, Application_mode=Gel.1-Umum, GDP=0.32, Inflation=1.4%, dll.) "
        "dan tidak mempengaruhi hasil secara signifikan.",
        icon="ℹ️",
    )
    st.divider()

    inputs = render_input_form()

    st.divider()
    _, col_btn, _ = st.columns([2, 3, 2])
    with col_btn:
        predict_btn = st.button(
            "🔍 Prediksi Risiko Dropout", use_container_width=True, type="primary"
        )

    if predict_btn:
        logger.info(
            "Prediksi diminta — sem2_approved=%d, tuition=%s, debtor=%s",
            inputs["sem2_approved"], inputs["tuition_up_to_date"], inputs["debtor"],
        )
        input_dict = build_input_dict(**inputs)
        input_df = pd.DataFrame([input_dict])
        model = load_model()

        if model is None:
            st.error(
                "⚠️ **Model belum tersedia.** "
                "Jalankan `notebook.ipynb` terlebih dahulu, lalu restart aplikasi."
            )
            st.code("jupyter notebook notebook.ipynb", language="bash")
        else:
            render_prediction_result(model, input_df, inputs)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    render_sidebar()

    st.title("🎓 Jaya Jaya Institut — Dropout Early Warning System")
    st.divider()

    tab_ml, tab_pred = st.tabs([
        "🤖 Solusi Machine Learning",
        "🔍 Prediksi Risiko Dropout",
    ])

    with tab_ml:
        render_ml_solution()

    with tab_pred:
        render_prediction_tab()


if __name__ == "__main__":
    main()
