import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Panelin ürettiği trade log dosyası
TRADES_FILE = "auto_trades_log.parquet"

# Ana model bundle (şimdilik sadece yeri belli olsun diye)
MODEL_BUNDLE_PATH = "models/nasdaq_meta_optuna_cv_v2.pkl"

# Çıkacak kalibrasyon modeli
CALIBRATOR_OUT_PATH = "models/nasdaq_meta_calibrator_v1.pkl"


def label_trades_for_training(
    trades_df: pd.DataFrame,
    good_pnl: float = 30.0,
    bad_pnl: float = 0.0,
) -> pd.DataFrame:
    df_lab = trades_df.copy()

    df_lab["label_good"] = np.where(
        df_lab["pnl_points"] >= good_pnl,
        1,
        np.where(
            df_lab["pnl_points"] <= bad_pnl,
            0,
            np.nan,
        ),
    )

    df_lab = df_lab.dropna(subset=["label_good"])
    df_lab["label_good"] = df_lab["label_good"].astype(int)
    return df_lab


def main():
    if not os.path.exists(TRADES_FILE):
        print(f"[!] {TRADES_FILE} bulunamadı. Önce panelden Auto-Trade Lab ile trade üret.")
        return

    trades = pd.read_parquet(TRADES_FILE)
    if trades.empty:
        print("[!] Trade dosyası boş.")
        return

    # Label'la
    trades_labeled = label_trades_for_training(trades)
    if trades_labeled.empty:
        print("[!] Belirlenen good/bad PnL eşiklerine göre kullanılabilir trade kalmadı.")
        return

    # Özellik: Şimdilik sadece p_up ile kalibre edelim
    X = trades_labeled[["p_up"]].values  # shape: (n,1)
    y = trades_labeled["label_good"].values

    # Basit logistic regression kalibratörü
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(X, y)

    # Basit metrikler
    proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, (proba >= 0.5).astype(int))

    print(f"[i] Eğitim örnek sayısı: {len(y)}")
    print(f"[i] ROC-AUC: {auc:.3f}")
    print(f"[i] Accuracy (0.5 cut): {acc:.3f}")

    # Kaydet
    out_obj = {
        "calibrator": clf,
        "features": ["p_up"],
        "info": {
            "good_pnl": 30.0,
            "bad_pnl": 0.0,
            "train_samples": int(len(y)),
            "auc": float(auc),
            "acc": float(acc),
        },
    }
    os.makedirs(os.path.dirname(CALIBRATOR_OUT_PATH), exist_ok=True)
    joblib.dump(out_obj, CALIBRATOR_OUT_PATH)
    print(f"[+] Kalibrasyon modeli kaydedildi: {CALIBRATOR_OUT_PATH}")


if __name__ == "__main__":
    main()