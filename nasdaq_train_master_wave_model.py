#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
import logging
from typing import List, Tuple, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from xgboost import XGBClassifier

# Versiyon kontrolü
XGB_VERSION = xgb.__version__.split('.')[0]  # "2" veya "1"

# ======================= CONFIGURATION =======================
MASTER_PATH = "./staging/nasdaq_master_features_v1.parquet"
WAVE_PATH = "./staging/nasdaq_full_wave_v3.parquet"
MODEL_PATH = "./models/nasdaq_master_wave_xgb_v2_robust.pkl"
SCALER_PATH = "./models/nasdaq_master_wave_scaler_v2_robust.pkl"
FEATS_PATH = "./models/nasdaq_master_wave_features_v2_robust.pkl"
ENCODERS_PATH = "./models/nasdaq_master_wave_encoders_v2_robust.pkl"
REPORT_PATH = "./models/training_report_v2_robust.txt"

# Label kolonu - EĞER merge sonrası adı değişirse otomatik ayarlanacak
TARGET_LABEL = "signal_wave"

# Wave'den alınacak kolonlar
WAVE_COLS_TO_MERGE = [
    'signal_wave', 'signal_wave_label',
    'up_move_pips', 'down_move_pips',
    'up_duration_bars', 'down_duration_bars',
    'wave_strength_pips', 'wave_duration_bars'
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./models/training_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================= ROBUST DATA LOADER =======================
class RobustDataLoader:
    def __init__(self, master_path: str, wave_path: str, wave_cols: List[str], target_label: str):
        self.master_path = master_path
        self.wave_path = wave_path
        self.wave_cols = wave_cols
        self.target_label = target_label
        self.actual_label_col = None  # Merge sonrası gerçek kolon adı
    
    def load(self) -> pd.DataFrame:
        """Robust bir şekilde veri yükle ve merge yap."""
        logger.info("=" * 80)
        logger.info("🚀 ROBUST DATA LOADING BAŞLIYOR")
        logger.info("=" * 80)
        
        # 1) Veri yükle
        master = pd.read_parquet(self.master_path)
        wave = pd.read_parquet(self.wave_path)
        
        logger.info(f"✅ Master loaded: {master.shape}")
        logger.info(f"✅ Wave loaded: {wave.shape}")
        
        # 2) Timestamp temizleme
        master = self._clean_timestamps(master, "Master")
        wave = self._clean_timestamps(wave, "Wave")
        
        # 3) Çakışan kolonları temizle
        master = self._remove_duplicate_columns(master, wave)
        
        # 4) Merge
        merged = self._safe_merge(master, wave)
        
        # 5) Label kolonunu doğrula ve ayarla
        self._validate_and_set_label_column(merged)
        
        # 6) NaN temizleme
        merged = merged.dropna(subset=[self.actual_label_col])
        
        logger.info(f"✅ Final merged shape: {merged.shape}")
        logger.info(f"✅ Label kolonu: {self.actual_label_col}")
        
        return merged
    
    def _clean_timestamps(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Timestamp kolonunu temizle ve standardize et."""
        logger.info(f"\n🔍 {name} timestamp temizleme...")
        
        if 'timestamp' not in df.columns:
            raise ValueError(f"'timestamp' kolonu {name} dosyasında yok!")
        
        # Convert ve timezone'u kaldır
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_localize(None)
        
        # NaN timestamp'leri düşür
        before = len(df)
        df = df.dropna(subset=['timestamp'])
        after = len(df)
        
        logger.info(f"   {name}: {before:,} -> {after:,} satır (NaN timestamp düştü)")
        
        return df
    
    def _remove_duplicate_columns(self, master: pd.DataFrame, wave: pd.DataFrame) -> pd.DataFrame:
        """Master'daki wave kolonları ile çakışanları düşür."""
        logger.info("\n🔍 Çakışan kolon kontrolü...")
        
        # Çakışan kolonları bul (timestamp hariç)
        duplicate_cols = [c for c in self.wave_cols if c in master.columns and c != 'timestamp']
        
        if duplicate_cols:
            logger.warning(f"⚠️  Çakışan {len(duplicate_cols)} kolon düşürülüyor: {duplicate_cols}")
            master = master.drop(columns=duplicate_cols)
        else:
            logger.info("✅ Çakışan kolon yok")
        
        return master
    
    def _safe_merge(self, master: pd.DataFrame, wave: pd.DataFrame) -> pd.DataFrame:
        """Güvenli merge işlemi."""
        logger.info("\n🔀 Merge işlemi...")
        
        # Wave'den sadece gerekli kolonları al
        wave_subset = wave[['timestamp'] + self.wave_cols].copy()
        
        # Inner merge
        merged = master.merge(wave_subset, on='timestamp', how='inner')
        
        logger.info(f"   Merge sonucu: {merged.shape}")
        
        # Otomatik suffix eklenmiş kolonları tespit et
        suffixes = ['_y', '_wave', '_right']
        for suffix in suffixes:
            if f"{self.target_label}{suffix}" in merged.columns:
                logger.warning(f"⚠️  Kolon adı değişti: {self.target_label} -> {self.target_label}{suffix}")
                # Gerçek kolon adını ayarla
                self.actual_label_col = f"{self.target_label}{suffix}"
                break
        else:
            self.actual_label_col = self.target_label
        
        return merged
    
    def _validate_and_set_label_column(self, merged: pd.DataFrame):
        """Label kolonunu valide et ve ayarla."""
        logger.info("\n🔍 Label kolonu validasyonu...")
        
        if self.actual_label_col not in merged.columns:
            logger.error(f"❌ Label kolonu '{self.actual_label_col}' bulunamadı!")
            logger.error(f"Mevcut kolonlar: {list(merged.columns)}")
            raise KeyError(f"Label kolonu merge sonrası kayboldu!")
        
        # NaN kontrolü
        nan_pct = merged[self.actual_label_col].isnull().mean() * 100
        logger.info(f"   NaN oranı: %{nan_pct:.2f}")
        
        # Sınıf dağılımı
        class_dist = merged[self.actual_label_col].value_counts().sort_index()
        logger.info(f"   Sınıf dağılımı:\n{class_dist}")

# ======================= FEATURE ENGINEERING =======================
def engineer_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Özellik mühendisliği - Wave özellikleri geçmiş bilgiyle (LEAK-FREE)."""
    logger.info("\n🔧 Özellik mühendisliği...")
    
    df = df.copy()
    original_cols = df.columns.tolist()
    
    # Timestamp'e göre sırala
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in [label_col, 'timestamp']]
    
    # ========== WAVE FEATURE ENGINEERING (LEAK-FREE) ==========
    logger.info("   🌊 Wave özellikleri oluşturuluyor (geçmiş bilgiyle - LEAK-FREE)...")
    
    # ⚠️ ÖNEMLİ: label_col (signal_wave) kullanarak feature üretme!
    # Sadece shift(1) ve öncesi kullanılabilir - current bar'ın label'ı kullanılamaz
    
    # 1) Geçmiş wave istatistikleri (lag-based, leak-free)
    wave_stats_cols = ['up_move_pips', 'down_move_pips', 'wave_strength_pips', 
                       'wave_duration_bars', 'up_duration_bars', 'down_duration_bars']
    
    for col in wave_stats_cols:
        if col in df.columns:
            # Rolling window ile geçmiş wave istatistikleri (shift(1) ile leak-free)
            for window in [10, 20, 50]:
                df[f'{col}_rolling_mean_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_max_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).max()
                df[f'{col}_rolling_std_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).std()
            
            # Son wave'in değeri (lag 1 ile - bir önceki wave)
            df[f'{col}_prev_wave'] = df[col].shift(1)
    
    # 2) Wave pattern özellikleri (SADECE geçmiş bilgiyle - label_col kullanmadan)
    if label_col in df.columns:
        # Önceki bar'ların label'larını kullan (shift(1), shift(2))
        prev_label = df[label_col].shift(1)
        prev_prev_label = df[label_col].shift(2)
        
        # Geçmiş wave başlangıç sinyalleri (önceki bar'da değişim oldu mu?)
        df['prev_wave_start'] = (prev_label != prev_prev_label).astype(int)
        df['prev_wave_start_up'] = ((prev_label == 1) & (df['prev_wave_start'] == 1)).astype(int)
        df['prev_wave_start_down'] = ((prev_label == 2) & (df['prev_wave_start'] == 1)).astype(int)
        
        # Son N bar içinde kaç wave başlangıcı oldu (geçmiş bilgiyle)
        df['wave_starts_last_5'] = df['prev_wave_start'].shift(1).rolling(window=5, min_periods=1).sum()
        df['wave_starts_last_20'] = df['prev_wave_start'].shift(1).rolling(window=20, min_periods=1).sum()
        
        # Son wave'in yönü (lag ile)
        df['prev_wave_direction'] = prev_label
        df['prev_prev_wave_direction'] = prev_prev_label
        
        # Geçmiş wave sürekliliği (önceki bar'da aynı yönde miydi?)
        df['prev_wave_continuity'] = (prev_label == prev_prev_label).astype(int)
        
        # Son wave'den bu yana geçen bar sayısı (geçmiş bilgiyle)
        # Önceki bar'ların wave başlangıçlarını kullanarak hesapla
        wave_start_mask = df['prev_wave_start'] == 1
        df['bars_since_last_wave'] = wave_start_mask.groupby((wave_start_mask).cumsum()).cumcount()
    
    # 4) Interaction features
    if 'atr14_M30' in df.columns and 'Volume_M30' in df.columns:
        df['atr_x_volume_M30'] = df['atr14_M30'] * df['Volume_M30']
    
    # 5) Lag features (teknik göstergeler için)
    for lag in [1, 2, 3, 5]:
        for col in numeric_cols[:15]:  # İlk 15 kolon için
            if col not in wave_stats_cols:  # Wave kolonlarını tekrar lag'leme
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # NaN temizleme
    df = df.dropna()
    
    logger.info(f"   ✅ Yeni kolonlar eklendi: {len(df.columns) - len(original_cols)}")
    logger.info(f"   ✅ Yeni shape: {df.shape}")
    
    return df

# ======================= MODEL PIPELINE =======================
class TradingModelPipeline:
    def __init__(self, data_loader: RobustDataLoader):
        self.data_loader = data_loader
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.label_col = None
    
    def run(self):
        """Tüm pipeline'ı çalıştır."""
        try:
            # 1) Load data
            merged = self.data_loader.load()
            self.label_col = self.data_loader.actual_label_col
            
            # 2) Feature engineering
            merged = engineer_features(merged, self.label_col)
            
            # 3) Leak detection
            leak_cols = self._detect_leak_columns(merged)
            
            # 4) Split & Scale
            splits = self._split_and_scale(merged, leak_cols)
            X_train, y_train, X_val, y_val, X_test, y_test = splits
            
            # 5) Train model
            self.model = self._train_model(X_train, y_train, X_val, y_val)
            
            # 6) Evaluate
            self._evaluate_model(self.model, X_val, y_val, X_test, y_test)
            
            # 7) Save artifacts
            self._save_artifacts()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ PIPELINE BAŞARIYLA TAMAMLANDI")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ PIPELINE HATASI: {e}", exc_info=True)
            raise
    
    def _detect_leak_columns(self, df: pd.DataFrame) -> List[str]:
        """Agresif sızıntı kolon tespiti - Tüm leak kaynakları."""
        logger.info("\n🔍 Sızıntı kolon tespiti (agresif mod)...")
        
        leak_cols = []
        
        # 1) Future/impact kolonları
        leak_patterns = ['future', 'fut_', 'impact_dir']
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in leak_patterns):
                leak_cols.append(col)
        
        # 2) Wave outcome kolonları - ORİJİNAL HALLERİ HER ZAMAN LEAK
        wave_outcome_cols = ['up_move_pips', 'down_move_pips', 'wave_strength_pips',
                            'wave_duration_bars', 'up_duration_bars', 'down_duration_bars']
        for col in wave_outcome_cols:
            if col in df.columns:
                leak_cols.append(col)  # Her durumda leak - outcome bilgisi
        
        # 3) Label kolonları ve türevleri
        # signal_wave_label gibi text label kolonları
        label_like_cols = [c for c in df.columns 
                          if 'label' in c.lower() 
                          or c == 'signal_wave_label'
                          or (c != self.label_col and 'signal_wave' in c.lower())]
        leak_cols.extend(label_like_cols)
        
        # 4) Current bar'dan label türetilen feature'lar
        # wave_start, wave_continuity gibi current signal_wave'den türetilenler
        label_derived_patterns = ['wave_start', 'wave_continuity']
        for col in df.columns:
            col_lower = col.lower()
            # Eğer "prev_" veya "rolling" içermiyorsa ve label türevi ise leak
            if any(pattern in col_lower for pattern in label_derived_patterns):
                if 'prev_' not in col_lower and 'rolling' not in col_lower:
                    leak_cols.append(col)
        
        # 5) Label kolonunu ekle (zaten drop edilecek ama listede olsun)
        if self.label_col not in leak_cols:
            leak_cols.append(self.label_col)
        
        # Duplicate'leri temizle
        leak_cols = list(set(leak_cols))
        
        logger.info(f"   ❌ Tespit edilen leak kolonlar: {len(leak_cols)}")
        logger.info(f"   📋 İlk 15: {leak_cols[:15]}")
        return leak_cols
    
    def _split_and_scale(self, df: pd.DataFrame, leak_cols: List[str]) -> Tuple:
        """Zamansal split ve ölçekleme - String/Label kolonları encode edilerek dahil edilir."""
        logger.info("\n📊 Train/Val/Test split...")
        
        # Drop listesi (zaman, leak ve label kolonları)
        time_cols = ['timestamp', 'datetime', 'date', 'open_time', 'close_time']
        
        # Label kolonları ve türevleri (explicit)
        label_like_cols = [c for c in df.columns 
                          if 'label' in c.lower() 
                          or c == 'signal_wave_label'
                          or (c != self.label_col and 'signal_wave' in c.lower())]
        
        drop_cols = [self.label_col] + leak_cols + time_cols + label_like_cols
        drop_cols = list(set([c for c in drop_cols if c in df.columns]))  # Duplicate temizle
        
        logger.info(f"   🗑️  Drop edilecek kolonlar: {len(drop_cols)}")
        logger.info(f"   📋 İlk 10: {drop_cols[:10]}")
        
        # X'i oluştur (drop edilenler hariç)
        X = df.drop(columns=drop_cols)
        y = df[self.label_col].astype(int)
        
        # Kolonları kategorilere ayır
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        string_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
        
        logger.info(f"   📊 Numeric kolonlar: {len(numeric_cols)}")
        logger.info(f"   📊 String kolonlar: {len(string_cols)}")
        if string_cols:
            logger.info(f"   📋 String kolonlar: {string_cols[:10]}")
        
        # String kolonları encode et
        X_encoded = X.copy()
        self.label_encoders = {}
        
        for col in string_cols:
            le = LabelEncoder()
            # NaN değerleri 'UNKNOWN' ile doldur
            X_encoded[col] = X_encoded[col].fillna('UNKNOWN').astype(str)
            X_encoded[col] = le.fit_transform(X_encoded[col])
            self.label_encoders[col] = le
            logger.info(f"   ✅ {col} encode edildi: {len(le.classes_)} unique değer")
        
        # Artık tüm kolonlar numeric
        X_final = X_encoded.select_dtypes(include=[np.number])
        
        self.feature_names = list(X_final.columns)
        
        logger.info(f"   ✅ Toplam feature sayısı: {len(self.feature_names)}")
        logger.info(f"   ✅ Çıkarılan kolonlar: {len(drop_cols)} (time/leak/label)")
        
        # Zaman bazlı sıralama
        df_sorted = df.sort_values('timestamp')
        X_final = X_final.loc[df_sorted.index].reset_index(drop=True)
        y = y.loc[df_sorted.index].reset_index(drop=True)
        
        # Split (60-20-20)
        n = len(X_final)
        train_end = int(n * 0.60)
        val_end = int(n * 0.80)
        
        X_train, y_train = X_final.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X_final.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X_final.iloc[val_end:], y.iloc[val_end:]
        
        logger.info(f"   Train: {X_train.shape[0]:,}")
        logger.info(f"   Val:   {X_val.shape[0]:,}")
        logger.info(f"   Test:  {X_test.shape[0]:,}")
        
        # Scale (tüm numeric kolonlar için)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    
    def _train_model(self, X_train, y_train, X_val, y_val) -> XGBClassifier:
        """XGBoost modeli eğit (XGBoost 2.x+ uyumlu)."""
        logger.info("\n🚂 Model eğitimi...")
        
        # Class weights hesapla ve kullan
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        logger.info(f"⚖️  Class weights: {class_weight_dict}")
        
        # Sample weight oluştur (XGBoost multi-class için gerekli)
        sample_weight_train = pd.Series(y_train).map(class_weight_dict).values
        
        # Model tanımlaması - XGBoost 2.0+ için early_stopping_rounds constructor'da
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes),
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            early_stopping_rounds=50  # ✅ XGBoost 2.0+ için constructor'da
        )
        
        # Early stopping ile fit (sample_weight ile)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight_train,  # Class imbalance için
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        logger.info(f"✅ Optimal iterasyon: {model.best_iteration}")
        logger.info(f"✅ Final best score: {model.best_score}")
        
        return model
    def _evaluate_model(self, model, X_val, y_val, X_test, y_test):
        """Modeli değerlendir."""
        logger.info("\n📈 Değerlendirme...")
        
        # Validation
        val_proba = model.predict_proba(X_val)
        val_logloss = log_loss(y_val, val_proba)
        logger.info(f"   Val logloss: {val_logloss:.4f}")
        
        # Test
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        test_logloss = log_loss(y_test, y_proba)
        
        logger.info(f"\n📊 Test LogLoss: {test_logloss:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, digits=3)}")
        
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"📊 Confusion Matrix:\n{cm}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n🔝 Top 10 Features:\n{importance_df.head(10)}")
        
        # Rapor kaydet
        with open(REPORT_PATH, 'w') as f:
            f.write("NASDAQ MASTER WAVE MODEL REPORT v2.0\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Validation LogLoss: {val_logloss:.4f}\n")
            f.write(f"Test LogLoss: {test_logloss:.4f}\n\n")
            f.write("Top 20 Features:\n")
            f.write(importance_df.head(20).to_string())
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))
        
        logger.info(f"💾 Rapor kaydedildi: {REPORT_PATH}")
    
    def _save_artifacts(self):
        """Model artifaklarını kaydet."""
        logger.info("\n💾 Kaydetme işlemi...")
        
        Path("./models").mkdir(exist_ok=True)
        
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.feature_names, FEATS_PATH)
        
        # Label encoder'ları kaydet (varsa)
        if hasattr(self, 'label_encoders') and self.label_encoders:
            joblib.dump(self.label_encoders, ENCODERS_PATH)
            logger.info(f"✅ Encoders: {ENCODERS_PATH} ({len(self.label_encoders)} encoder)")
        else:
            logger.info("ℹ️  Label encoder yok (string kolon yok)")
        
        logger.info(f"✅ Model: {MODEL_PATH}")
        logger.info(f"✅ Scaler: {SCALER_PATH}")
        logger.info(f"✅ Features: {FEATS_PATH}")

# ======================= MAIN =======================
def main():
    warnings.filterwarnings('ignore')
    
    # Robust loader oluştur
    loader = RobustDataLoader(
        master_path=MASTER_PATH,
        wave_path=WAVE_PATH,
        wave_cols=WAVE_COLS_TO_MERGE,
        target_label=TARGET_LABEL
    )
    
    # Pipeline oluştur ve çalıştır
    pipeline = TradingModelPipeline(data_loader=loader)
    pipeline.run()

if __name__ == "__main__":
    main()