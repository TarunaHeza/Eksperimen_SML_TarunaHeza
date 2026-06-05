"""
automate_TarunaHeza.py
======================
Script otomatisasi preprocessing dataset MBG (Makan Bergizi Gratis).
Menjalankan seluruh pipeline preprocessing dari data mentah hingga data siap latih.

Penggunaan:
    python automate_TarunaHeza.py
    python automate_TarunaHeza.py --input path/ke/raw.csv --output path/ke/output/
"""

import argparse
import os
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings('ignore')

# ─── Coba import NLTK, fallback jika tidak ada ───────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARN] NLTK tidak tersedia. Tokenisasi menggunakan split sederhana.")


# ─── Konstanta ────────────────────────────────────────────────────────────────
POSITIVE_WORDS = [
    'bagus', 'baik', 'manfaat', 'bermanfaat', 'berhasil', 'sukses',
    'mendukung', 'dukung', 'apresiasi', 'positif', 'senang', 'gizi',
    'nutrisi', 'sehat', 'meningkat', 'membantu', 'gratis', 'alhamdulillah',
    'luar biasa', 'hebat', 'bagus sekali', 'program bagus'
]

NEGATIVE_WORDS = [
    'buruk', 'gagal', 'masalah', 'bermasalah', 'korupsi', 'menyedihkan',
    'kecewa', 'kelaparan', 'tidak merata', 'basi', 'mual', 'sakit',
    'kontroversi', 'kritik', 'tolak', 'menolak', 'pecat', 'ditunda',
    'terlambat', 'tidak tepat', 'merugikan'
]

ADDITIONAL_STOPWORDS = {
    'yg', 'dgn', 'utk', 'krn', 'jg', 'sdh', 'dr', 'dlm', 'tsb',
    'nya', 'kita', 'ini', 'itu', 'juga', 'ada', 'dan', 'yang',
    'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'adalah'
}

ENGAGEMENT_COLS = ['favorite_count', 'retweet_count', 'reply_count', 'quote_count']


# ─── Fungsi Preprocessing ─────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    print(f"[1/9] Memuat data dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Shape awal: {df.shape}")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Memilih kolom yang relevan dan filter bahasa Indonesia."""
    print("[2/9] Seleksi kolom & filter bahasa Indonesia...")
    cols = ['text', 'favorite_count', 'retweet_count',
            'reply_count', 'quote_count', 'lang']
    df = df[cols].copy()
    df = df[df['lang'] == 'in'].drop(columns=['lang'])
    print(f"      Shape setelah filter: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat berdasarkan kolom teks."""
    print("[3/9] Menghapus duplikat...")
    before = len(df)
    df = df.drop_duplicates(subset='text').reset_index(drop=True)
    print(f"      Dihapus: {before - len(df)} baris | Sisa: {len(df)}")
    return df


def clean_tweet(text: str) -> str:
    """
    Membersihkan teks tweet:
    - Hapus mention, URL, simbol hashtag
    - Hapus karakter non-huruf
    - Lowercase & strip whitespace
    """
    text = re.sub(r'@\w+', '', text)           # hapus mention
    text = re.sub(r'http\S+|www\S+', '', text)  # hapus URL
    text = re.sub(r'#', '', text)               # hapus simbol hashtag
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # hapus non-huruf
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Menerapkan clean_tweet ke kolom teks."""
    print("[4/9] Pembersihan teks (mention, URL, karakter khusus)...")
    df['text_clean'] = df['text'].apply(clean_tweet)
    before = len(df)
    df = df[df['text_clean'].str.len() >= 10].reset_index(drop=True)
    print(f"      Teks terlalu pendek dihapus: {before - len(df)} baris")
    return df


def tokenize_remove_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    """Tokenisasi dan penghapusan stopwords Bahasa Indonesia."""
    print("[5/9] Tokenisasi & hapus stopwords...")

    if NLTK_AVAILABLE:
        stop_words = set(stopwords.words('indonesian'))
    else:
        stop_words = set()

    stop_words.update(ADDITIONAL_STOPWORDS)

    def process(text):
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        return ' '.join(tokens)

    df['text_processed'] = df['text_clean'].apply(process)
    return df


def handle_outliers(df: pd.DataFrame,
                    cols: list = ENGAGEMENT_COLS,
                    factor: float = 3.0) -> pd.DataFrame:
    """Menangani outlier dengan metode IQR capping (Winsorization)."""
    print("[6/9] Penanganan outlier dengan IQR capping...")
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"      {col}: {n_outlier} outlier di-cap [{lower:.1f}, {upper:.1f}]")
    return df


def normalize_features(df: pd.DataFrame,
                        cols: list = ENGAGEMENT_COLS) -> pd.DataFrame:
    """Normalisasi fitur numerik menggunakan MinMaxScaler."""
    print("[7/9] Normalisasi fitur numerik (MinMaxScaler)...")
    valid_cols = [c for c in cols if c in df.columns]
    scaler = MinMaxScaler()
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    return df


def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Labeling sentimen semi-otomatis berbasis keyword."""
    print("[8/9] Labeling sentimen (keyword-based)...")

    def assign_label(text):
        text_lower = text.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
        if pos > neg:
            return 'positif'
        elif neg > pos:
            return 'negatif'
        return 'netral'

    df['label'] = df['text'].apply(assign_label)

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    print(f"      Distribusi label:\n{df['label'].value_counts().to_string()}")
    return df


def save_output(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """Menyimpan hasil preprocessing ke file CSV."""
    print(f"[9/9] Menyimpan output ke: {output_dir}/{filename}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"      Berhasil! Shape final: {df.shape}")
    return output_path


# ─── Pipeline Utama ───────────────────────────────────────────────────────────

def run_preprocessing(input_path: str,
                      output_dir: str = 'data_mbg_preprocessing',
                      output_file: str = 'data_mbg_preprocessing.csv') -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan.

    Parameters
    ----------
    input_path  : str  - Path ke file CSV mentah
    output_dir  : str  - Direktori output
    output_file : str  - Nama file output

    Returns
    -------
    pd.DataFrame - Dataset yang sudah diproses
    """
    print("\n" + "=" * 55)
    print("  PIPELINE PREPROCESSING DATA MBG")
    print("=" * 55)

    df = load_data(input_path)
    df = select_columns(df)
    df = remove_duplicates(df)
    df = clean_text_column(df)
    df = tokenize_remove_stopwords(df)
    df = handle_outliers(df)
    df = normalize_features(df)
    df = label_sentiment(df)
    output_path = save_output(df, output_dir, output_file)

    print("\n" + "=" * 55)
    print(f"  PREPROCESSING SELESAI")
    print(f"  Output: {output_path}")
    print("=" * 55 + "\n")

    return df


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automate preprocessing dataset MBG'
    )
    parser.add_argument(
        '--input',
        default='data_mbg_raw.csv',
        help='Path ke file CSV mentah (default: data_mbg_raw.csv)'
    )
    parser.add_argument(
        '--output',
        default='data_mbg_preprocessing',
        help='Direktori output (default: data_mbg_preprocessing/)'
    )
    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input,
        output_dir=args.output,
        output_file='data_mbg_preprocessing.csv'
    )
