import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import MinMaxScaler
import joblib

st.set_page_config(page_title="Prediksi Diabetes dengan SVM", layout="wide")
st.title("🩺 Prediksi Penyakit Diabetes dengan SVM")
st.markdown("""
Aplikasi ini memuat analisis data, visualisasi, pelatihan model, dan prediksi real-time terhadap kemungkinan seseorang terkena diabetes.
""")

# -------------------------------------------
# Upload Dataset
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data berhasil diunggah.")
else:
    st.warning("⚠️ Silakan unggah file `diabetes.csv` untuk melanjutkan.")
    st.stop()

# -------------------------------------------
# 1. Data Cleaning
df = df.dropna()
st.subheader("📄 Data Setelah Dibersihkan")
st.dataframe(df.head())

# 2. Normalisasi Data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df.drop("Outcome", axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled["Outcome"] = df["Outcome"]

# 3. Split Data
X = df_scaled.drop("Outcome", axis=1)
y = df_scaled["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=30)

# 4. Train Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.subheader("📊 Evaluasi Model")
st.write(f"**Akurasi:** {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# 6. Confusion Matrix
st.subheader("📉 Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Tidak Diabetes", "Diabetes"],
            yticklabels=["Tidak Diabetes", "Diabetes"], ax=ax_cm)
st.pyplot(fig_cm)

# 7. Visualisasi Perbandingan Aktual vs Prediksi
st.subheader("📈 Perbandingan Aktual vs Prediksi")
fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
ax_pred.plot(y_test.values, 'b-', label='Aktual')
ax_pred.plot(y_pred, 'r--', label='Prediksi')
ax_pred.legend()
ax_pred.set_title("Perbandingan Aktual vs Prediksi")
ax_pred.set_xlabel("Index Sampel")
ax_pred.set_ylabel("Kelas")
st.pyplot(fig_pred)

# 8. Visualisasi Sebaran Fitur
st.subheader("📌 Visualisasi Distribusi Fitur")
compare_df = X_test.copy()
compare_df["Actual"] = y_test.values
compare_df["Predicted"] = y_pred

selected_feature = st.selectbox("Pilih fitur", X.columns)
fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=compare_df, x=selected_feature, hue="Actual", fill=True, common_norm=False, palette="Blues", alpha=0.5, linewidth=2, ax=ax_dist)
sns.kdeplot(data=compare_df, x=selected_feature, hue="Predicted", linestyle='--', common_norm=False, palette="Reds", linewidth=2, ax=ax_dist)
ax_dist.set_title(f"Sebaran {selected_feature} berdasarkan Kelas Aktual dan Prediksi")
st.pyplot(fig_dist)

# 9. Simpan Model
joblib.dump(model, "svm_model.pkl")

# 10. Prediksi Berdasarkan Input User
st.subheader("🧪 Prediksi Data Pasien Baru (Realtime)")
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

input_data = []
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    val = cols[i % 2].number_input(f"{feature}", min_value=0.0, step=0.1)
    input_data.append(val)

if st.button("Prediksi Diabetes"):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_normalized = scaler.transform(input_df)
    prediction = model.predict(input_normalized)

    if prediction[0] == 1:
        st.error("🔴 Hasil Prediksi: Pasien **mengalami Diabetes**.")
    else:
        st.success("🟢 Hasil Prediksi: Pasien **tidak mengalami Diabetes**.")
