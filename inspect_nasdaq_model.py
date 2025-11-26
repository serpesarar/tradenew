import joblib

MODEL_PATH = "nasdaq_30m_ULTIMATE_v1.pkl"

obj = joblib.load(MODEL_PATH)

print("Tip:", type(obj))

if isinstance(obj, dict):
    print("\nBu dosya bir dict, içindeki anahtarlar:")
    for k in obj.keys():
        print(" -", k)
else:
    print("\nBu dosya dict değil, şu tip bir nesne:")
    print(obj)