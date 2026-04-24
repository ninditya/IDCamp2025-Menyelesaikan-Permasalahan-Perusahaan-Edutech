"""
Utility functions untuk proyek Prediksi Dropout Mahasiswa Jaya Jaya Institut.

Modul ini berisi fungsi-fungsi pembantu yang digunakan bersama antara
notebook.ipynb dan app.py, mencakup preprocessing, deteksi outlier,
evaluasi model, dan penyimpanan artifact.

Author: Ninditya
Email: ninditya.sna025@gmail.com
"""

# Standard library
import logging
import os
from pathlib import Path
from typing import Optional

# Third-party
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, shapiro
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_dataset(url: str, sep: str = ";") -> pd.DataFrame:
    """Memuat dataset dari URL atau path lokal.

    Args:
        url (str): URL atau path file CSV.
        sep (str): Delimiter CSV. Default ";".

    Returns:
        pd.DataFrame: Dataset yang telah dimuat.

    Raises:
        ValueError: Jika dataset kosong setelah dimuat.
    """
    df = pd.read_csv(url, sep=sep)
    if df.empty:
        raise ValueError(f"Dataset kosong setelah dimuat dari {url}")
    logger.info("Dataset dimuat: %d baris, %d kolom", df.shape[0], df.shape[1])
    return df


# ============================================================================
# EDA HELPERS
# ============================================================================
def detect_outliers_iqr(data: pd.DataFrame, col: str) -> tuple:
    """Mendeteksi outlier pada sebuah kolom menggunakan metode IQR.

    Args:
        data (pd.DataFrame): DataFrame yang akan dianalisis.
        col (str): Nama kolom numerik yang akan dicek.

    Returns:
        tuple: (outlier_count, outlier_pct, lower_bound, upper_bound)
            - outlier_count (int): Jumlah baris yang termasuk outlier.
            - outlier_pct (float): Persentase outlier dari total baris.
            - lower_bound (float): Batas bawah IQR.
            - upper_bound (float): Batas atas IQR.
    """
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (data[col] < lower) | (data[col] > upper)
    count = int(outlier_mask.sum())
    pct = count / len(data) * 100
    return count, pct, lower, upper


def print_outlier_summary(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Mencetak ringkasan outlier untuk semua kolom numerik.

    Args:
        df (pd.DataFrame): Dataset yang akan dianalisis.
        numeric_cols (list): Daftar nama kolom numerik.

    Returns:
        pd.DataFrame: DataFrame berisi ringkasan outlier per kolom.
    """
    records = []
    print("Outlier Analysis (IQR Method):")
    print("-" * 90)
    for col in numeric_cols:
        count, pct, low, up = detect_outliers_iqr(df, col)
        print(f"{col:45s}: {count:4d} outliers ({pct:.2f}%) | range: [{low:.2f}, {up:.2f}]")
        records.append({"feature": col, "outlier_count": count,
                        "outlier_pct": pct, "lower_bound": low, "upper_bound": up})
    return pd.DataFrame(records)


def chi_square_test(df: pd.DataFrame, categorical_cols: list,
                    target_col: str = "Status") -> pd.DataFrame:
    """Melakukan Chi-Square test antara fitur kategorikal dan target.

    Args:
        df (pd.DataFrame): Dataset yang akan dianalisis.
        categorical_cols (list): Daftar nama kolom kategorikal.
        target_col (str): Nama kolom target. Default "Status".

    Returns:
        pd.DataFrame: Hasil Chi-Square test (feature, chi2, p_value, significant).
    """
    results = []
    print("Chi-Square Test Results:")
    print("-" * 65)
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p, dof, _ = chi2_contingency(contingency)
        significant = p < 0.05
        label = "✓ Significant" if significant else "✗ Not Significant"
        print(f"{col:30s}: chi2={chi2:8.2f}, p={p:.4f} → {label}")
        results.append({
            "feature": col,
            "chi2": chi2,
            "p_value": p,
            "significant": significant,
        })
    return pd.DataFrame(results)


def shapiro_wilk_test(df: pd.DataFrame, cols: list,
                      sample_size: int = 5000) -> pd.DataFrame:
    """Melakukan Shapiro-Wilk normality test untuk kolom numerik.

    Menggunakan sampling jika jumlah baris melebihi sample_size karena
    Shapiro-Wilk tidak efisien untuk n > 5000.

    Args:
        df (pd.DataFrame): Dataset yang akan dianalisis.
        cols (list): Daftar nama kolom yang akan diuji.
        sample_size (int): Maksimum sampel untuk Shapiro-Wilk. Default 5000.

    Returns:
        pd.DataFrame: Hasil test (feature, statistic, p_value, is_normal).
    """
    results = []
    print("Shapiro-Wilk Normality Test:")
    print("-" * 60)
    for col in cols:
        sample = df[col].dropna().sample(min(sample_size, len(df)), random_state=42)
        stat, p = shapiro(sample)
        is_normal = p > 0.05
        label = "Normal" if is_normal else "Not Normal"
        print(f"{col:40s}: stat={stat:.4f}, p={p:.4f} → {label}")
        results.append({"feature": col, "statistic": stat, "p_value": p, "is_normal": is_normal})
    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_dropout_rate_by_category(df: pd.DataFrame, col: str,
                                  target_col: str = "Status",
                                  dropout_label: str = "Dropout",
                                  figsize: tuple = (10, 5),
                                  output_path: Optional[str] = None) -> None:
    """Membuat bar chart dropout rate per kategori.

    Args:
        df (pd.DataFrame): Dataset yang akan divisualisasikan.
        col (str): Nama kolom kategori yang akan dianalisis.
        target_col (str): Nama kolom target. Default "Status".
        dropout_label (str): Label kelas dropout dalam kolom target. Default "Dropout".
        figsize (tuple): Ukuran figure. Default (10, 5).
        output_path (str, optional): Path untuk menyimpan gambar. Jika None, hanya tampil.
    """
    dropout_rate = (
        df.groupby(col)[target_col]
        .apply(lambda x: (x == dropout_label).mean() * 100)
        .reset_index()
        .rename(columns={target_col: "dropout_rate"})
        .sort_values("dropout_rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        dropout_rate[col].astype(str),
        dropout_rate["dropout_rate"],
        color=["#DC2626" if r > 50 else "#1E3A8A" for r in dropout_rate["dropout_rate"]],
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title(f"Dropout Rate per {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("Dropout Rate (%)", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Plot disimpan ke %s", output_path)
    plt.show()


# ============================================================================
# MODEL EVALUATION
# ============================================================================
def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = "Model") -> dict:
    """Mengevaluasi performa model dengan metrik lengkap.

    Args:
        model: Trained sklearn estimator atau pipeline.
        x_test (np.ndarray): Fitur data test.
        y_test (np.ndarray): Label data test.
        model_name (str): Nama model untuk label output. Default "Model".

    Returns:
        dict: Dictionary berisi accuracy, roc_auc, dan classification_report.
    """
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("%s — Accuracy: %.4f, ROC-AUC: %.4f", model_name, acc, auc)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  F1 (dropout): {report['1']['f1-score']:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return {"accuracy": acc, "roc_auc": auc, "report": report}


# ============================================================================
# ARTIFACT I/O
# ============================================================================
def save_artifacts(model, feature_names: list,
                   output_dir: str = "model") -> None:
    """Menyimpan model pipeline dan nama fitur ke disk.

    Args:
        model: Trained sklearn pipeline.
        feature_names (list): Daftar nama fitur yang digunakan model.
        output_dir (str): Direktori output. Default "model".
    """
    os.makedirs(output_dir, exist_ok=True)

    pipeline_path = Path(output_dir) / "pipeline.pkl"
    features_path = Path(output_dir) / "feature_names.pkl"

    joblib.dump(model, pipeline_path)
    joblib.dump(feature_names, features_path)

    logger.info("Pipeline disimpan ke %s", pipeline_path)
    logger.info("Feature names disimpan ke %s", features_path)


def load_artifacts(model_dir: str = "model") -> tuple:
    """Memuat model pipeline dan nama fitur dari disk.

    Args:
        model_dir (str): Direktori tempat artifact tersimpan. Default "model".

    Returns:
        tuple: (pipeline, feature_names)
            - pipeline: Trained sklearn pipeline.
            - feature_names (list): Daftar nama fitur.

    Raises:
        FileNotFoundError: Jika file artifact tidak ditemukan.
    """
    pipeline_path = Path(model_dir) / "pipeline.pkl"
    features_path = Path(model_dir) / "feature_names.pkl"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline tidak ditemukan: {pipeline_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature names tidak ditemukan: {features_path}")

    pipeline = joblib.load(pipeline_path)
    feature_names = joblib.load(features_path)

    logger.info("Artifacts berhasil dimuat dari %s", model_dir)
    return pipeline, feature_names
