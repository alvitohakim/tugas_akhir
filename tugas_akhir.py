import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Data akurasi per-model
data = {
    'Model': ['Sequential Sederhana', 'BiLSTM tanpa Attention', 'BiLSTM + Self-Attention SMALL', 'BiLSTM + Multi-Head Attention'],
    'Split 1': [0.7, 0.6875, 0.675, 0.675],
    'Split 2': [0.6962025316455697, 0.7721518987341772, 0.8227848101265823, 0.7721518987341772],
    'Split 3': [0.5949367088607594, 0.6962025316455697, 0.6780865794939671, 0.6329113924055633]
}

# Mengubah data menjadi DataFrame untuk tampilan tabel
df = pd.DataFrame(data)

# Judul halaman
st.title('Perbandingan Akurasi Model Sentimen')

# Deskripsi
st.write("""
Aplikasi ini menampilkan perbandingan akurasi antara empat model deep learning dalam tugas analisis sentimen.
Data ini diambil dari eksperimen yang menggunakan dataset bola versi lama dengan tiga split data.
""")

# Menampilkan tabel
st.subheader('Tabel Hasil Akurasi Model')
st.write(df)

# Membuat Grafik
st.subheader('Grafik Perbandingan Akurasi Model')

# Membuat plot perbandingan akurasi per-model
plt.figure(figsize=(10, 6))
for model in df['Model']:
    plt.plot(df.columns[1:], df.loc[df['Model'] == model, df.columns[1:]].values.flatten(), marker='o', label=model)

plt.title('Perbandingan Akurasi Model per Split')
plt.xlabel('Split Data')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Menampilkan grafik di Streamlit
st.pyplot(plt)

