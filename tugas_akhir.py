import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Data akurasi per-model untuk semua dataset
data = {
    'Model': ['Sequential Sederhana', 'BiLSTM tanpa Attention', 'BiLSTM + Self-Attention SMALL', 'BiLSTM + Multi-Head Attention'],
    'IMDb Akurasi': [0.8797, 0.8732, 0.8671, 0.8794],  # Satu set akurasi untuk IMDb
    'Bola Lama Split 1': [0.7, 0.6875, 0.675, 0.675],
    'Bola Lama Split 2': [0.6962025316455697, 0.7721518987341772, 0.8227848101265823, 0.7721518987341772],
    'Bola Lama Split 3': [0.5949367088607594, 0.6962025316455697, 0.6780865794939671, 0.6329113924055633],
    'Bola Baru Split 1': [0.8645, 0.9291, 0.9451, 0.9482],
    'Bola Baru Split 2': [0.8748, 0.9402, 0.9474, 0.9633],
    'Bola Baru Split 3': [0.8748, 0.9402, 0.9474, 0.9633]
}

# Mengubah data menjadi DataFrame
df = pd.DataFrame(data)

# Judul halaman
st.title('Perbandingan Akurasi Model Sentimen')

# Deskripsi
st.write("""
Aplikasi ini menampilkan perbandingan akurasi antara empat model deep learning dalam tugas analisis sentimen
menggunakan tiga dataset yang berbeda: IMDb, dataset bola versi lama, dan dataset bola versi baru. 
Pilih model dan dataset untuk melihat hasil akurasi lebih detail.
""")

# Dropdown untuk memilih dataset dan model
dataset_option = st.selectbox('Pilih Dataset', ['IMDb', 'Bola Lama', 'Bola Baru'])
model_option = st.selectbox('Pilih Model', df['Model'].tolist())

# Menampilkan hasil akurasi untuk model yang dipilih
if dataset_option == 'IMDb':
    splits = ['IMDb Akurasi']  # Hanya satu set akurasi untuk IMDb
elif dataset_option == 'Bola Lama':
    splits = ['Bola Lama Split 1', 'Bola Lama Split 2', 'Bola Lama Split 3']
else:
    splits = ['Bola Baru Split 1', 'Bola Baru Split 2', 'Bola Baru Split 3']

# Filter hasil berdasarkan model dan dataset
selected_model_data = df[df['Model'] == model_option]

# Menampilkan hasil akurasi untuk model yang dipilih dan dataset
st.subheader(f'Hasil Akurasi untuk Model: {model_option} pada Dataset {dataset_option}')
st.write(f'Akurasi pada split data:')
split_accuracies = selected_model_data[splits].values.flatten()
for i, accuracy in enumerate(split_accuracies):
    st.write(f'{splits[i]}: {accuracy}')

# Menampilkan Tabel Hasil Akurasi Semua Model
st.subheader('Tabel Hasil Akurasi Semua Model per Dataset')
st.write(df)

# Membuat Grafik untuk perbandingan
st.subheader('Grafik Perbandingan Akurasi Model')
plt.figure(figsize=(10, 6))

# Plot semua model
for model in df['Model']:
    if dataset_option == 'IMDb':
        accuracies = df.loc[df['Model'] == model, ['IMDb Akurasi']].values.flatten()
    elif dataset_option == 'Bola Lama':
        accuracies = df.loc[df['Model'] == model, ['Bola Lama Split 1', 'Bola Lama Split 2', 'Bola Lama Split 3']].values.flatten()
    else:
        accuracies = df.loc[df['Model'] == model, ['Bola Baru Split 1', 'Bola Baru Split 2', 'Bola Baru Split 3']].values.flatten()

    plt.plot(splits, accuracies, marker='o', label=model)

# Menambahkan label, judul, dan legend
plt.title(f'Perbandingan Akurasi Model pada Dataset {dataset_option}')
plt.xlabel('Split Data')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Menampilkan grafik di Streamlit
st.pyplot(plt)
