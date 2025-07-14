# 1. Import Library
from sklearn.neighbors import KNeighborsClassifier

# 2. Data
data = [
    [0.3, 7.21, 'Douglas Fir'],
    [0.18, 5.12, 'Douglas Fir'],
    [0.46, 8.83, 'Douglas Fir'],
    [0.63, 12.08, 'Douglas Fir'],
    [0.23, 5.81, 'Douglas Fir'],
    [0.56, 13.5, 'Douglas Fir'],
    [0.39, 10.9, 'Douglas Fir'],
    [0.41, 6.79, 'Douglas Fir'],
    [0.62, 10.66, 'Douglas Fir'],
    [0.43, 10.5, 'Douglas Fir'],
    [0.15, 2.67, 'Douglas Fir'],
    [0.19, 20.34, 'White Pine'],
    [0.17, 19.72, 'White Pine'],
    [0.17, 19.8, 'White Pine'],
    [0.22, 23.7, 'White Pine'],
    [0.45, 32.51, 'White Pine'],
    [0.39, 26.23, 'White Pine'],
    [0.42, 32.51, 'White Pine'],
    [0.38, 29.18, 'White Pine'],
    [0.3, 26.1, 'White Pine'],
    [0.18, 21.51, 'White Pine'],
]

# 3. Sistem Pakar + Certainty Factor
def expert_system_predict(diameter, tinggi):
    if diameter > 0.25 and tinggi > 20:
        return "White Pine", 0.85
    elif diameter <= 0.25 and tinggi <= 20:
        return "Douglas Fir", 0.75
    elif diameter > 0.25 and tinggi <= 20:
        return "Douglas Fir", 0.60
    elif diameter <= 0.25 and tinggi > 20:
        return "White Pine", 0.60
    else:
        return "Unknown", 0.5


# 4. Siapkan data untuk ML
X = [[d[0], d[1]] for d in data]
y = [d[2] for d in data]

# Melatih model KNN pada semua data ok
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# COMPUTER VISION MODULE (DUMMY, USER INPUT) SOAL NO 10
def detect_tree_features_from_image(image_path):
    print(f"\n[Modul Vision] Memproses gambar: {image_path} (dummy input, vision belum diimplementasikan).")
    detected_diameter = float(input("Masukkan diameter hasil deteksi vision (misal 0.32): "))
    detected_tinggi = float(input("Masukkan tinggi hasil deteksi vision (misal 25.5): "))
    return detected_diameter, detected_tinggi

print("=== DEMO COMPUTER VISION ===")
image_path = input("Masukkan nama file gambar pohon (misal pohon1.jpg): ")
diameter, tinggi = detect_tree_features_from_image(image_path)

pakar_pred, cf = expert_system_predict(diameter, tinggi)
knn_pred = knn.predict([[diameter, tinggi]])[0]

print("\nHasil Prediksi Gambar:")
print(f"Sistem Pakar: {pakar_pred} (Certainty Factor: {cf})")
print(f"KNN         : {knn_pred}")

if pakar_pred == knn_pred:
    print("Kedua metode menghasilkan prediksi yang sama.")
else:
    print("Prediksi antara Sistem Pakar dan KNN berbeda.")
#==============================================================================

# 5. Evaluasi kombinasi
hasil_project = []
jumlah_benar = 0
total_nilai_kebenaran = 0

for diameter, tinggi, label_asli in data:
    pred_pakar, cf = expert_system_predict(diameter, tinggi) #sistem pakar

    pred_knn = knn.predict([[diameter, tinggi]])[0]  # KNN

    kebenaran = cf if pred_pakar == pred_knn else 0   # Penilaian
    benar = pred_pakar == pred_knn
    hasil_project.append({
        "diameter": diameter,
        "tinggi": tinggi,
        "pakar": pred_pakar,
        "cf": cf,
        "knn": pred_knn,
        "benar": benar,
        "nilai_kebenaran": kebenaran
    })
    total_nilai_kebenaran += kebenaran
    if benar:
        jumlah_benar += 1

rata2_nilai_kebenaran = total_nilai_kebenaran / len(data)
akurasi_kecocokan = jumlah_benar / len(data)

# Bagian User Input Interaktif
print("\n=== Cek Prediksi Berdasarkan Hasil Input ===")
try:
    diameter_input = float(input("Masukkan diameter batang pohon (contoh 0.22): "))
    tinggi_input = float(input("Masukkan tinggi pohon (contoh 21.5): "))

    # Prediksi Sistem Pakar
    pred_pakar, cf = expert_system_predict(diameter_input, tinggi_input)
    # Prediksi KNN
    pred_knn = knn.predict([[diameter_input, tinggi_input]])[0]

    print(f"\nHasil Prediksi:")
    print(f"Sistem Pakar: {pred_pakar} (Certainty Factor: {cf})")
    print(f"KNN         : {pred_knn}")

    if pred_pakar == pred_knn:
        print("Kedua metode menghasilkan prediksi yang sama.")
    else:
        print("Prediksi antara Sistem Pakar dan KNN berbeda.")
except Exception as e:
    print(f"Input tidak valid: {e}")

print("\nHasil Penilaian Sistem Pakar vs KNN :\n")
for h in hasil_project:
    print(f"Diameter: {h['diameter']:.2f}, Tinggi: {h['tinggi']:.2f} | Sistem Pakar: {h['pakar']} (CF={h['cf']}) | KNN: {h['knn']} | {'Cocok' if h['benar'] else 'Tidak Cocok'} | Nilai Kebenaran: {h['nilai_kebenaran']}")

print(f"\nRata-rata nilai kebenaran : {rata2_nilai_kebenaran:.3f}")
print(f"Akurasi kecocokan sistem pakar dengan KNN: {akurasi_kecocokan*100:.1f}%")
