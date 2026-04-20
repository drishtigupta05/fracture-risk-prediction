"""Test the API with the honest model (CNN+TBT, no BMD)."""
import requests, json

url   = "http://127.0.0.1:5000/predict"
files = {"file": open("FinallDATA/images/0032793634.png", "rb")}
data  = {"age": "65", "height": "155", "is_menopausal": "1"}

print("[TEST] Sending prediction request to honest model...")
r = requests.post(url, files=files, data=data, timeout=120)
d = r.json()

print(f"Status : {r.status_code}")
print(f"Success: {d.get('success')}")
print()
print("=== Fracture Risk ===")
print(f"  Level : {d.get('fracture_risk')}")
print(f"  Score : {d.get('risk_score')}")
print(f"  Color : {d.get('risk_color')}")
print()
print("=== Classification ===")
print(f"  Prediction : {d.get('prediction')}")
print(f"  Confidence : {d.get('confidence')}")
print(f"  Probs      : {d.get('probabilities')}")
print()
print("=== Preprocessing ===")
print(f"  Steps: {len(d.get('preprocessing_images',{}).get('steps',[]))}")
print()
if not d.get("success"):
    print("ERROR:", d.get("error"))
