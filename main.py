import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------
# Konfigurasi Halaman Aplikasi
# ---------------------------------
st.set_page_config(
    page_title="Prediksi Customer Churn",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Fungsi Pemuatan Model
# ---------------------------------
@st.cache_resource
def load_model():
    """Memuat pipeline model dari file .pkl yang sudah disimpan."""
    try:
        model = joblib.load('churn_logreg_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("File model 'churn_logreg_pipeline.pkl' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan app.py")
        return None

# Memuat model di awal
model = load_model()

# ---------------------------------
# Antarmuka Sidebar untuk Input Data
# ---------------------------------
st.sidebar.header('Input Data Pelanggan')

def user_input_features():
    """Membuat widget input di sidebar dan mengembalikan DataFrame."""
    # Data Demografi
    gender = st.sidebar.selectbox('Jenis Kelamin', ('Male', 'Female'))
    partner = st.sidebar.selectbox('Memiliki Partner?', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Memiliki Tanggungan?', ('Yes', 'No'))

    # Data Akun
    tenure = st.sidebar.slider('Lama Berlangganan (Bulan)', 1, 72, 12)
    contract = st.sidebar.selectbox('Jenis Kontrak', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Tagihan Paperless?', ('Yes', 'No'))
    payment_method = st.sidebar.selectbox('Metode Pembayaran', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.sidebar.slider('Tagihan Bulanan ($)', 18.0, 120.0, 50.0)
    total_charges = st.sidebar.slider('Total Tagihan ($)', 18.0, 9000.0, 1000.0)

    # Data Layanan Telepon
    phone_service = st.sidebar.selectbox('Layanan Telepon?', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines?', ('Yes', 'No', 'No phone service'))

    # Data Layanan Internet
    internet_service = st.sidebar.selectbox('Layanan Internet', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Keamanan Online', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Backup Online', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Proteksi Perangkat', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Dukungan Teknis', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('TV Streaming', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Film Streaming', ('Yes', 'No', 'No internet service'))

    # Membuat dictionary dari input
    data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Mengubah dictionary menjadi DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ---------------------------------
# Tampilan Halaman Utama
# ---------------------------------
st.title('Prediksi Customer Churn 📞')
st.markdown("""
Aplikasi ini memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan (churn) dari perusahaan telekomunikasi. 
Masukkan data pelanggan di sidebar kiri dan klik tombol **Prediksi** untuk melihat hasilnya.
""")
st.markdown("---")

# Menampilkan data input yang dimasukkan pengguna
st.subheader('Data Pelanggan yang Dimasukkan:')
st.dataframe(input_df.T.rename(columns={0: 'Nilai'}))

# Tombol Prediksi diletakkan di bawah data input di sidebar
if st.sidebar.button('Prediksi'):
    if model is not None:
        # Melakukan prediksi menggunakan pipeline yang sudah dimuat
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Menampilkan hasil dengan tampilan yang lebih baik
        st.markdown("---")
        st.subheader('Hasil Prediksi Model')

        if prediction[0] == 1:
            st.error('**🚨 Pelanggan ini Berisiko Tinggi untuk Churn**', icon="🚨")
        else:
            st.success('**✅ Pelanggan ini Kemungkinan Besar akan Tetap Bertahan**', icon="✅")

        # Menampilkan probabilitas dalam bentuk kolom metrik
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Probabilitas Bertahan", value=f"{prediction_proba[0][0]*100:.2f}%")
        with col2:
            st.metric(label="Probabilitas Churn", value=f"{prediction_proba[0][1]*100:.2f}%")
    else:
        st.warning("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini dibuat sebagai bagian dari studi kasus Data Science untuk memprediksi customer churn.")