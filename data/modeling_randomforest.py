import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 PREDIKSI HARGA PROPERTI - RANDOM FOREST")
print("   4 MODEL: Fisik | +Makro | +Lokasi | +Semua")
print("="*80)

# ============================================================
# 1. LOAD DATA PROPERTI
# ============================================================
print("\n[1] Load data properti...")
df_properti = pd.read_csv('rumah123_cleans8_final.csv')
print(f"    Data properti: {len(df_properti)} baris")

# Ekstrak tanggal
df_properti['tanggal'] = pd.to_datetime(df_properti['tanggal_clean'])
df_properti['tahun'] = df_properti['tanggal'].dt.year
df_properti['bulan'] = df_properti['tanggal'].dt.month

print(f"    Range tanggal properti: {df_properti['tanggal'].min().strftime('%d %B %Y')} - {df_properti['tanggal'].max().strftime('%d %B %Y')}")
print(f"    Bulan unik di properti: {sorted(df_properti[['tahun', 'bulan']].drop_duplicates().values.tolist())}")

# ============================================================
# 2. LOAD DATA EKONOMI DARI FILE EXCEL HASIL MERGE
# ============================================================
print("\n[2] Load data ekonomi dari file Excel...")

# Baca file Excel hasil merge
df_ekonomi = pd.read_excel('Data_Ekonomi_Properti_Oct2025_Apr2026.xlsx', 
                            sheet_name='Data Mentah')

print(f"    Kolom tersedia: {df_ekonomi.columns.tolist()}")
print(f"    Data ekonomi: {len(df_ekonomi)} bulan")
print(f"    Range: {df_ekonomi['tahun'].min()}-{int(df_ekonomi['bulan'].min()):02d} sampai {df_ekonomi['tahun'].max()}-{int(df_ekonomi['bulan'].max()):02d}")

# Ambil kolom yang dibutuhkan, rename kurs_mean -> kurs_usd
df_ekonomi = df_ekonomi[['tahun', 'bulan', 'kurs_mean', 'inflasi_malang', 'inflasi_nasional']].copy()
df_ekonomi.rename(columns={'kurs_mean': 'kurs_usd'}, inplace=True)

print(f"\n    📊 Data ekonomi yang akan digabung:")
print(df_ekonomi.to_string(index=False))

# ============================================================
# 3. GABUNGKAN DATA EKONOMI DENGAN PROPERTI
# ============================================================
print("\n[3] Menggabungkan data ekonomi ke properti...")

# Merge berdasarkan tahun & bulan
df_properti = df_properti.merge(df_ekonomi, on=['tahun', 'bulan'], how='left')

# Cek missing values setelah merge
for col in ['kurs_usd', 'inflasi_malang', 'inflasi_nasional']:
    missing = df_properti[col].isnull().sum()
    if missing > 0:
        print(f"    ⚠️  {col} masih missing {missing} baris")
        # Cek bulan apa yang missing
        missing_bulan = df_properti[df_properti[col].isnull()][['tahun', 'bulan']].drop_duplicates()
        print(f"        Bulan missing: {missing_bulan.values.tolist()}")
        # Isi dengan data terdekat (backward fill atau forward fill)
        df_properti[col] = df_properti[col].fillna(method='ffill').fillna(method='bfill')
        print(f"        Diisi dengan forward/backward fill")

print(f"    ✅ Data properti final: {len(df_properti)} baris")
print(f"\n    📊 Statistik Data Makro:")
print(f"    Kurs USD/IDR      : Rp {df_properti['kurs_usd'].min():,.0f} - Rp {df_properti['kurs_usd'].max():,.0f}")
print(f"    Inflasi Malang    : {df_properti['inflasi_malang'].min():.2f}% - {df_properti['inflasi_malang'].max():.2f}%")
print(f"    Inflasi Nasional  : {df_properti['inflasi_nasional'].min():.2f}% - {df_properti['inflasi_nasional'].max():.2f}%")

# Sample data
print(f"\n    Sample data properti + ekonomi:")
sample_cols = ['tahun', 'bulan', 'kecamatan', 'harga_bersih', 'kurs_usd', 'inflasi_malang', 'inflasi_nasional']
print(df_properti[sample_cols].head(10).to_string())

# Cek distribusi data per bulan
print(f"\n    Jumlah properti per bulan:")
print(df_properti.groupby(['tahun', 'bulan']).size().to_string())

# ============================================================
# 4. ENCODE KECAMATAN
# ============================================================
print("\n[4] Mengencode kecamatan...")
le = LabelEncoder()
df_properti['kode_kecamatan'] = le.fit_transform(df_properti['kecamatan'])
print(f"    Jumlah kecamatan: {len(le.classes_)}")
print(f"    Daftar: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ============================================================
# 5. BERSIHKAN DATA
# ============================================================
print("\n[5] Membersihkan data...")
df_properti['harga_per_m2'] = df_properti['harga_bersih'] / df_properti['luas_tanah_m2']
batas_bawah = df_properti['harga_per_m2'].quantile(0.03)
batas_atas = df_properti['harga_per_m2'].quantile(0.97)
df = df_properti[(df_properti['harga_per_m2'] >= batas_bawah) & (df_properti['harga_per_m2'] <= batas_atas)]
df = df.dropna()
print(f"    Data final: {len(df)} baris (setelah buang outlier)")

# ============================================================
# 6. LOG TRANSFORMASI HARGA
# ============================================================
df['log_harga'] = np.log(df['harga_bersih'])

# ============================================================
# 7. PISAH DATA TRAINING & TESTING
# ============================================================
print("\n[6] Split data training & testing...")

# Pisahkan dulu sebelum buat fitur kecamatan
X_base = df[['kamar_tidur', 'kamar_mandi', 'luas_tanah_m2', 'luas_bangunan',
             'kurs_usd', 'inflasi_malang', 'inflasi_nasional', 
             'kode_kecamatan', 'kecamatan', 'harga_bersih']].copy()
y = df['log_harga']

X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

print(f"    Data training: {len(X_train)} baris")
print(f"    Data testing: {len(X_test)} baris")

# ============================================================
# 8. HITUNG FITUR KECAMATAN DARI DATA TRAINING SAJA
# ============================================================
print("\n[7] Menghitung fitur kecamatan (dari training only)...")

# Gabungkan training dengan target asli untuk hitung statistik
train_data = X_train.copy()
train_data['harga_asli'] = np.exp(y_train)

# Hitung statistik per kecamatan HANYA dari data training
harga_mean_kec = train_data.groupby('kecamatan')['harga_asli'].mean()
harga_median_kec = train_data.groupby('kecamatan')['harga_asli'].median()
harga_std_kec = train_data.groupby('kecamatan')['harga_asli'].std()

# Klasifikasi kecamatan berdasarkan median (dari training)
median_all = harga_median_kec.median()
q75_all = harga_median_kec.quantile(0.75)
q25_all = harga_median_kec.quantile(0.25)

def klasifikasi_kec(median_harga):
    if median_harga >= q75_all:
        return 3  # Premium
    elif median_harga >= median_all:
        return 2  # Menengah Atas
    elif median_harga >= q25_all:
        return 1  # Menengah
    else:
        return 0  # Ekonomis

kelas_kec = harga_median_kec.apply(klasifikasi_kec)

# Fungsi untuk apply fitur kecamatan
def apply_kec_features(data, prefix=''):
    data = data.copy()
    data['kec_mean'] = data['kecamatan'].map(harga_mean_kec)
    data['kec_median'] = data['kecamatan'].map(harga_median_kec)
    data['kec_std'] = data['kecamatan'].map(harga_std_kec)
    data['kec_kelas'] = data['kecamatan'].map(kelas_kec)
    
    # Isi missing (kalau ada kecamatan di test yg tidak muncul di train)
    data['kec_mean'] = data['kec_mean'].fillna(train_data['harga_asli'].median())
    data['kec_median'] = data['kec_median'].fillna(train_data['harga_asli'].median())
    data['kec_std'] = data['kec_std'].fillna(0)
    data['kec_kelas'] = data['kec_kelas'].fillna(1).astype(int)
    
    return data

# Apply ke training dan testing
X_train = apply_kec_features(X_train)
X_test = apply_kec_features(X_test)

# Tampilkan profil kecamatan
print("\n    📊 Profil Kecamatan (dari data training):")
print("    " + "="*70)
print(f"    {'Kecamatan':<22} {'Median':<15} {'Kelas':<12} {'Jumlah':<10}")
print("    " + "-"*70)
kelas_label = ['Ekonomis', 'Menengah', 'Menengah Atas', 'Premium']
for kec in harga_median_kec.sort_values(ascending=False).index:
    jml = len(train_data[train_data['kecamatan'] == kec])
    kls = kelas_label[int(kelas_kec[kec])]
    print(f"    {kec:<22} Rp{harga_median_kec[kec]:>10,.0f}  {kls:<12} {jml:<10}")

# ============================================================
# 9. DAFTAR FITUR UNTUK 4 MODEL
# ============================================================
print("\n" + "="*60)
print("📋 4 MODEL YANG AKAN DILATIH")
print("="*60)

fitur_fisik = ['kamar_tidur', 'kamar_mandi', 'luas_tanah_m2', 'luas_bangunan']
fitur_makro = ['kurs_usd', 'inflasi_malang', 'inflasi_nasional']
fitur_lokasi = ['kode_kecamatan', 'kec_mean', 'kec_median', 'kec_std', 'kec_kelas']

models_config = [
    {'name': '1. Fisik Saja', 'features': fitur_fisik},
    {'name': '2. + Makro', 'features': fitur_fisik + fitur_makro},
    {'name': '3. + Lokasi', 'features': fitur_fisik + fitur_lokasi},
    {'name': '4. + Semua', 'features': fitur_fisik + fitur_makro + fitur_lokasi}
]

for i, m in enumerate(models_config):
    print(f"   Model {i}: {m['name']:<20} -> {len(m['features'])} fitur")

# ============================================================
# 10. TRAINING 4 MODEL
# ============================================================
print("\n" + "="*60)
print("🚀 MELATIH 4 MODEL...")
print("="*60)

all_models = []
results = []

for i, config in enumerate(models_config):
    print(f"\n[{i+1}/4] Training: {config['name']}")
    
    # Ambil fitur dari X_train dan X_test
    X_tr = X_train[config['features']]
    X_te = X_test[config['features']]
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_train)
    
    y_pred_log = model.predict(X_te)
    y_pred = np.exp(y_pred_log)
    y_actual = np.exp(y_test)
    
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
    
    results.append({
        'name': config['name'],
        'n_features': len(config['features']),
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    })
    
    all_models.append(model)
    
    print(f"    ✅ R² = {r2:.4f} ({r2*100:.2f}%)")
    print(f"    ✅ MAPE = {mape:.2%}")

# ============================================================
# 11. SIMPAN SEMUA 4 MODEL
# ============================================================
print("\n📦 MENYIMPAN MODEL...")

os.makedirs('models', exist_ok=True)

# Simpan 4 model
for i, model_obj in enumerate(all_models):
    with open(f'models/model_{i}.pkl', 'wb') as f:
        pickle.dump(model_obj, f)
    print(f"   ✅ Model {i} saved: models/model_{i}.pkl")

# Simpan encoder
with open('models/encoder_kecamatan.pkl', 'wb') as f:
    pickle.dump(le, f)
print(f"   ✅ Encoder saved: models/encoder_kecamatan.pkl")

# Simpan metrics
with open('models/results_optimasi.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"   ✅ Metrics saved: models/results_optimasi.pkl")

# Simpan model terbaik
best_idx = 0
best_r2 = 0
for i, res in enumerate(results):
    if res['r2'] > best_r2:
        best_r2 = res['r2']
        best_idx = i

with open('models/model_terbaik.pkl', 'wb') as f:
    pickle.dump(all_models[best_idx], f)
print(f"   ✅ Best model saved: models/model_terbaik.pkl (Model {best_idx})")

# Simpan statistik kecamatan untuk prediksi nanti
statistik_kec = {
    'harga_mean': harga_mean_kec.to_dict(),
    'harga_median': harga_median_kec.to_dict(),
    'harga_std': harga_std_kec.to_dict(),
    'kelas': kelas_kec.to_dict()
}
with open('models/statistik_kecamatan.pkl', 'wb') as f:
    pickle.dump(statistik_kec, f)
print(f"   ✅ Statistik kecamatan saved: models/statistik_kecamatan.pkl")

print("\n✅ SEMUA 4 MODEL TERSIMPAN DI FOLDER 'models/'")
print("   - model_0.pkl (Fisik Saja)")
print("   - model_1.pkl (+ Makro)")
print("   - model_2.pkl (+ Lokasi)")
print("   - model_3.pkl (+ Semua)")
print("   - encoder_kecamatan.pkl")
print("   - statistik_kecamatan.pkl")
print("   - results_optimasi.pkl")

# Ringkasan
print("\n" + "="*80)
print("📊 RINGKASAN PERFORMA MODEL")
print("="*80)
for res in results:
    print(f"{res['name']:<20} | R²: {res['r2']:.4f} | MAE: Rp{res['mae']:,.0f} | MAPE: {res['mape']:.2%}")

# Feature importance model terbaik
print(f"\n📊 Feature Importance Model Terbaik (Model {best_idx}):")
importance = pd.DataFrame({
    'feature': models_config[best_idx]['features'],
    'importance': all_models[best_idx].feature_importances_
}).sort_values('importance', ascending=False)
for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"   {row['feature']:<25} {row['importance']:.4f} {bar}")
    
print("="*50)
print("🔍 DIAGNOSA DATA")
print("="*50)

# Cek dulu kolom yang tersedia
print(f"\n📋 Kolom di df: {df.columns.tolist()}")

# 1. Range harga
print(f"\n1. Range Harga:")
print(f"   Min: Rp {df['harga_bersih'].min():,.0f}")
print(f"   Max: Rp {df['harga_bersih'].max():,.0f}")
print(f"   Mean: Rp {df['harga_bersih'].mean():,.0f}")
print(f"   Median: Rp {df['harga_bersih'].median():,.0f}")
print(f"   Std: Rp {df['harga_bersih'].std():,.0f}")

# 2. Cek duplikat dari data PROPETI (sebelum dibersihkan)
print(f"\n2. Duplikasi Data:")
print(f"   Total data (setelah bersih): {len(df)}")

# Baca ulang data properti untuk cek duplikat
df_asli = pd.read_csv('rumah123_cleans5_final.csv')
print(f"   Total data asli: {len(df_asli)}")
if 'deskripsi' in df_asli.columns:
    print(f"   Unique deskripsi: {df_asli['deskripsi'].nunique()}")
    dup_count = len(df_asli) - df_asli['deskripsi'].nunique()
    print(f"   Duplikat: {dup_count} ({dup_count/len(df_asli)*100:.1f}%)")
elif 'judul' in df_asli.columns:
    print(f"   Unique judul: {df_asli['judul'].nunique()}")
    dup_count = len(df_asli) - df_asli['judul'].nunique()
    print(f"   Duplikat (judul): {dup_count} ({dup_count/len(df_asli)*100:.1f}%)")

# 3. Cek anomali
if 'luas_tanah_m2' in df.columns and 'kamar_tidur' in df.columns:
    df['rasio'] = df['kamar_tidur'] / df['luas_tanah_m2'] * 100
    anomali = len(df[df['rasio'] > 10])
    print(f"\n3. Anomali (>10 KT per 100m2): {anomali} data")
    
    anomali2 = len(df[(df['luas_bangunan'] > df['luas_tanah_m2'] * 5)])
    print(f"   LB > 5x LT (tidak wajar): {anomali2} data")

# 4. Variasi makro
if 'kurs_usd' in df.columns:
    print(f"\n4. Variasi Data Makro:")
    print(f"   Kurs unique values: {df['kurs_usd'].nunique()}")
    print(f"   Kurs range: {df['kurs_usd'].min():.0f} - {df['kurs_usd'].max():.0f}")
if 'inflasi_malang' in df.columns:
    print(f"   Inflasi Malang unique: {df['inflasi_malang'].nunique()}")
    print(f"   Inflasi Malang range: {df['inflasi_malang'].min():.2f} - {df['inflasi_malang'].max():.2f}")
if 'inflasi_nasional' in df.columns:
    print(f"   Inflasi Nasional unique: {df['inflasi_nasional'].nunique()}")
    print(f"   Inflasi Nasional range: {df['inflasi_nasional'].min():.2f} - {df['inflasi_nasional'].max():.2f}")

# 5. Outlier per m2
if 'harga_per_m2' in df.columns:
    print(f"\n5. Harga per m2:")
    print(f"   Min: Rp {df['harga_per_m2'].min():,.0f}/m2")
    print(f"   1%: Rp {df['harga_per_m2'].quantile(0.01):,.0f}/m2")
    print(f"   5%: Rp {df['harga_per_m2'].quantile(0.05):,.0f}/m2")
    print(f"   Median: Rp {df['harga_per_m2'].median():,.0f}/m2")
    print(f"   95%: Rp {df['harga_per_m2'].quantile(0.95):,.0f}/m2")
    print(f"   99%: Rp {df['harga_per_m2'].quantile(0.99):,.0f}/m2")
    print(f"   Max: Rp {df['harga_per_m2'].max():,.0f}/m2")

# 6. Distribusi kecamatan
if 'kecamatan' in df.columns:
    print(f"\n6. Distribusi Kecamatan:")
    print(df['kecamatan'].value_counts().to_string())    