import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ---------------- SAFE STYLES ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(255,75,75,0.3);
    margin-bottom: 20px;
}
h1, h2, h3 {
    color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🛍️ Mall Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Creative ML App using KMeans</p>", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📥 Enter Customer Details")

income = st.number_input(
    "💰 Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=50,
    step=1
)

score = st.number_input(
    "🧾 Spending Score (1–100)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("✨ Predict Customer Group"):
    X = np.array([[income, score]])
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.success(f"🎯 Customer belongs to **Cluster {cluster}**")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(income, score, s=250)
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Position")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🚀 Built with Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)