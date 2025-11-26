# patterns_nasdaq.py
import numpy as np
import pandas as pd

def find_swings(df: pd.DataFrame, high_col="high", low_col="low", order: int = 3):
    """
    Basit swing tespiti:
      - swing_high_idx: lokal tepeler
      - swing_low_idx : lokal dipler
    """
    highs = df[high_col].to_numpy()
    lows  = df[low_col].to_numpy()
    n = len(df)

    swing_high_idx = []
    swing_low_idx = []

    for i in range(order, n - order):
        h = highs[i]
        l = lows[i]

        if h == np.max(highs[i - order : i + order + 1]):
            swing_high_idx.append(i)
        if l == np.min(lows[i - order : i + order + 1]):
            swing_low_idx.append(i)

    return np.array(swing_high_idx, dtype=int), np.array(swing_low_idx, dtype=int)


def detect_candlestick_patterns(df: pd.DataFrame, open_col="open", high_col="high",
                                low_col="low", close_col="close"):
    """
    Küçük mum formasyonları:
      - Hammer / Shooting Star
      - Bullish / Bearish Engulfing
    Son N mumda bulduklarını döner.
    """
    o = df[open_col].to_numpy()
    h = df[high_col].to_numpy()
    l = df[low_col].to_numpy()
    c = df[close_col].to_numpy()

    patterns = []

    def body(i): return abs(c[i] - o[i])
    def upper_wick(i): return h[i] - max(c[i], o[i])
    def lower_wick(i): return min(c[i], o[i]) - l[i]

    n = len(df)
    start_idx = max(0, n - 80)   # sadece son 80 muma bak

    for i in range(start_idx, n):
        if i == 0:
            continue

        b = body(i)
        if b == 0:
            continue

        uw = upper_wick(i)
        lw = lower_wick(i)

        # Hammer (dipte uzun alt fitil)
        if lw > 2 * b and uw < 0.3 * b and c[i] > o[i]:
            patterns.append({
                "index": i,
                "name": "Hammer",
                "direction": "up",
                "strength": float(min(lw / b, 5.0)),
            })

        # Shooting Star (tepede uzun üst fitil)
        if uw > 2 * b and lw < 0.3 * b and c[i] < o[i]:
            patterns.append({
                "index": i,
                "name": "Shooting Star",
                "direction": "down",
                "strength": float(min(uw / b, 5.0)),
            })

        # Bullish Engulfing
        if i >= 1:
            prev_bear = c[i - 1] < o[i - 1]
            curr_bull = c[i] > o[i]
            if prev_bear and curr_bull:
                if (c[i] >= o[i - 1]) and (o[i] <= c[i - 1]):
                    patterns.append({
                        "index": i,
                        "name": "Bullish Engulfing",
                        "direction": "up",
                        "strength": float(
                            abs(c[i] - o[i]) / (abs(c[i - 1] - o[i - 1]) + 1e-6)
                        ),
                    })

        # Bearish Engulfing
        if i >= 1:
            prev_bull = c[i - 1] > o[i - 1]
            curr_bear = c[i] < o[i]
            if prev_bull and curr_bear:
                if (o[i] >= c[i - 1]) and (c[i] <= o[i - 1]):
                    patterns.append({
                        "index": i,
                        "name": "Bearish Engulfing",
                        "direction": "down",
                        "strength": float(
                            abs(c[i] - o[i]) / (abs(c[i - 1] - o[i - 1]) + 1e-6)
                        ),
                    })

    # En güçlü ilk 5 tanesini döndür
    patterns.sort(key=lambda x: -x["strength"])
    return patterns[:5]


def detect_flag_pattern(df: pd.DataFrame, close_col="close", lookback_trend=150, lookback_flag=40):
    """
    Basit bayrak/flama tespiti:
      - Son 150 barda net yukarı/aşağı trend
      - Son 40 barda volatilite düşmüş = konsolidasyon
    Çıktı: dict veya None
    """
    if len(df) < lookback_trend + lookback_flag:
        return None

    closes = df[close_col].to_numpy(dtype=float)
    n = len(closes)

    recent_trend = closes[n - lookback_trend : n - lookback_flag]
    recent_flag  = closes[n - lookback_flag :]

    if len(recent_trend) < 10 or len(recent_flag) < 10:
        return None

    # Trend için lineer regresyon
    x_tr = np.arange(len(recent_trend))
    coef = np.polyfit(x_tr, recent_trend, 1)
    slope = coef[0]

    # Volatilite ölçüleri
    vol_trend = np.std(np.diff(recent_trend))
    vol_flag  = np.std(np.diff(recent_flag))

    if vol_trend == 0:
        return None

    vol_ratio = vol_flag / vol_trend  # < 1 ise daralan yapı

    last_price = float(closes[-1])

    # Basit: bayrak üst çizgisi = son 20 barın max'ı
    flag_high = float(np.max(recent_flag[-20:]))
    flag_low  = float(np.min(recent_flag[-20:]))

    info = {
        "type": None,
        "direction": None,
        "progress": 0.0,
        "distance_to_break": None,
        "flag_high": flag_high,
        "flag_low": flag_low,
        "trend_slope": slope,
        "vol_ratio": vol_ratio,
    }

    # Yukarı trend + daralan volatilite → bullish flag adayı
    if slope > 0 and vol_ratio < 0.8:
        info["type"] = "Bullish Flag"
        info["direction"] = "up"
        dist = flag_high - last_price
        info["distance_to_break"] = dist
        if flag_high != flag_low:
            pct = (last_price - flag_low) / (flag_high - flag_low)
            info["progress"] = float(min(max(pct, 0.0), 1.0))

    # Aşağı trend + daralan volatilite → bearish flag adayı
    elif slope < 0 and vol_ratio < 0.8:
        info["type"] = "Bearish Flag"
        info["direction"] = "down"
        dist = last_price - flag_low
        info["distance_to_break"] = dist
        if flag_high != flag_low:
            pct = (flag_high - last_price) / (flag_high - flag_low)
            info["progress"] = float(min(max(pct, 0.0), 1.0))
    else:
        return None

    return info