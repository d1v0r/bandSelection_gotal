import streamlit as st
import numpy as np
import torch
import scipy.io as sio
from model.autoencoder import AmortizedAESelector

# Učitaj trenirani model
device = "cuda" if torch.cuda.is_available() else "cpu"
B = 103  # broj bandova za PaviaU, promijeni prema potrebi!
K = 30   # ili koliko želiš prikazati

# Pretpostavljamo da imaš spremljene težine modela nakon treniranja
MODEL_PATH = "amortized_ae.pt"
model = AmortizedAESelector(num_bands=B).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

st.title("Optimalni bandovi na temelju hiperspektralne slike")

uploaded_file = st.file_uploader("Učitaj .mat ili .npy datoteku slike", type=["mat", "npy"])

if uploaded_file is not None:
    # Učitaj i pripremi podatke
    if uploaded_file.name.endswith(".mat"):
        mat = sio.loadmat(uploaded_file)
        if 'paviaU' in mat:
            img = mat['paviaU']
        else:
            st.error("Učitaj valjanu .mat datoteku s poljem 'paviaU'")
            st.stop()
    else:
        img = np.load(uploaded_file)

    # Provjera dimenzija i format
    if img.ndim == 3:
        H, W, B_ = img.shape
        assert B_ == B, "Broj bandova mora odgovarati modelu (" + str(B) + ")."
        img_2d = img.reshape(-1, B).astype(np.float32)
    elif img.ndim == 2 and img.shape[1] == B:
        img_2d = img
    else:
        st.error("Sliku očekujem kao [visina, širina, bandovi] ili [piks, bandovi]")
        st.stop()

    # Normalizacija (isti princip kao kod treniranja)
    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min())
    img_tensor = torch.from_numpy(img_2d).to(device)

    # Provedi kroz model i dohvati vektor importance
    with torch.no_grad():
        _, importance = model(img_tensor)
        importance_mean = importance.mean(dim=0).cpu().numpy()
        topk_idx = np.argsort(importance_mean)[-K:][::-1]

    st.write(f"### Top-{K} optimalnih bandova za ovu sliku (indeksi, od najvažnijeg):")
    st.write(topk_idx)

    st.bar_chart(importance_mean)

    st.write("""
    Bandovi sa najvećim vrijednostima importance su informacijski najreprezentativniji za ovu specifičnu sliku prema naučenoj globalnoj funkciji.
    """)
