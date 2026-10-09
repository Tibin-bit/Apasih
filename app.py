import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import re
import pandas as pd
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global AI E-Waste Detector Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi riwayat analisis
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# ---------------------------------------------------------
# 2. MANAJEMEN API KEY (AUTOMATIC SECRETS / SIDEBAR)
# ---------------------------------------------------------
api_key = ""
if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
    api_key = st.secrets["GEMINI_API_KEY"]

with st.sidebar:
    st.title("⚡ AI Core Settings")
    st.caption("Universal E-Waste Detection System")
    
    if api_key:
        st.success("🟢 API Key Otomatis Terhubung!")
    else:
        api_key = st.text_input("Masukkan Gemini API Key:", type="password", help="Dapatkan API Key gratis dari Google AI Studio")
    
    st.markdown("---")
    st.markdown("### 📋 Standar Klasifikasi PBB")
    st.info("Sistem ini mengacu pada pedoman **UN Global E-Waste Monitor** untuk 6 Kategori Sampah Elektronik Internasional.")
    st.markdown("---")
    st.caption("v5.0 Master Engine — Ultra Fast & Fail-Safe")

# ---------------------------------------------------------
# 3. FUNGSI ANALISIS DENGAN DYNAMIC AUTO-DISCOVERY & RAW DECODER
# ---------------------------------------------------------
def get_dynamic_models(key):
    """Mencari nama model aktif langsung dari server Google secara real-time."""
    active_models = []
    try:
        genai.configure(api_key=key)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                active_models.append(m.name)
    except Exception:
        pass

    # Daftar prioritas model Gemini Vision terbaru & stabil
    priority = [
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    
    # Gabungkan model terdeteksi dan prioritas tanpa duplikat
    final_list = []
    for p in priority:
        if p not in final_list:
            final_list.append(p)
    for a in active_models:
        if a not in final_list:
            final_list.append(a)
            
    return final_list

def analyze_ewaste_master(image, key):
    genai.configure(api_key=key)
    
    prompt = """
    Bertindaklah sebagai Ahli Pengolahan Sampah Elektronik (E-Waste Specialist) berstandar Internasional PBB.
    Analisis gambar ini secara mendalam untuk mengidentifikasi perangkat/komponen elektronik apa pun di dunia.
     Output WAJIB berupa data JSON MURNI tanpa teks pembuka atau penutup.

    Struktur JSON yang wajib dihasilkan:
    {
        "nama_objek": "Nama spesifik perangkat/komponen elektronik",
        "kategori_un": "Salah satu dari 6 Kategori UN E-Waste (1. Temperature Exchange Equipment, 2. Screens & Monitors, 3. Lamps, 4. Large Equipment, 5. Small Equipment, 6. Small IT & Telecommunication)",
        "deskripsi": "Deskripsi fungsi dan karakteristik perangkat",
        "tingkat_bahaya": "Tinggi / Sedang / Rendah",
        "skor_bahaya": 8,
        "bahan_berbahaya": ["Bahan 1", "Bahan 2", "Bahan 3"],
        "potensi_logam_mulia": {
            "Emas (Au)": "Ada / Tidak ada / Tinggi",
            "Perak (Ag)": "Ada / Sedang",
            "Tembaga (Cu)": "Ada / Tinggi"
        },
        "instruksi_penanganan": [
            "Langkah 1 penanganan aman",
            "Langkah 2 pemisahan komponen",
            "Langkah 3 opsi pembuangan/daur ulang"
        ],
        "dapat_didaur_ulang_persen": 80
    }
    """
    
    models_to_try = get_dynamic_models(key)
    gen_config = {"response_mime_type": "application/json"}
    
    response = None
    used_model = ""
    last_err = ""

    # Percobaan pemanggilan model secara aman (Fail-Safe Loop)
    for m_name in models_to_try:
        try:
            try:
                model = genai.GenerativeModel(m_name, generation_config=gen_config)
                res = model.generate_content([prompt, image])
            except Exception:
                model = genai.GenerativeModel(m_name)
                res = model.generate_content([prompt, image])
                
            if res and res.text:
                response = res
                used_model = m_name
                break
        except Exception as e:
            last_err = str(e)
            continue

    if response is None:
        return None, None, f"Semua model gagal merespons. Detail error: {last_err}"

    # Ekstraksi JSON Anti-Fail Menggunakan JSONDecoder().raw_decode()
    try:
        raw_text = response.text.strip()
        # Bersihkan tag markdown code block
        raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text).strip()
        
        start_idx = raw_text.find('{')
        if start_idx != -1:
            decoder = json.JSONDecoder()
            parsed_data, _ = decoder.raw_decode(raw_text[start_idx:])
            return parsed_data, used_model, None
        else:
            return None, used_model, "Respon AI tidak mengandung data JSON."
    except Exception as e:
        return None, used_model, f"Gagal membaca format JSON: {str(e)}"

# ---------------------------------------------------------
# 4. TAMPILAN UTAMA APLIKASI
# ---------------------------------------------------------
st.title("⚡ Global AI E-Waste Detector Pro")
st.markdown("Sistem Pengenal & Analisis Bahaya Sampah Elektronik Berbasis Vision AI Universal")

tab1, tab2 = st.tabs(["🔍 Analisis E-Waste", "📊 Dashboard & Riwayat"])

# --- TAB 1: ANALISIS GAMBAR ---
with tab1:
    col_input, col_output = st.columns([1, 1.2], gap="medium")
    
    with col_input:
        st.subheader("1. Pilih Sumber Gambar")
        source = st.radio("Metode Input:", ["Kamera Langsung 📷", "Unggah Berkas 📁"], horizontal=True)
        
        input_image = None
        if "Kamera" in source:
            cam_file = st.camera_input("Ambil Foto Perangkat E-Waste")
            if cam_file:
                input_image = Image.open(cam_file)
        else:
            uploaded_file = st.file_uploader("Pilih gambar perangkat (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                input_image = Image.open(uploaded_file)
                
        if input_image:
            st.image(input_image, caption="Gambar Siap Dianalisis", use_container_width=True)
            analyze_btn = st.button("🚀 Jalankan Analisis AI Universal", type="primary", use_container_width=True)

    with col_output:
        st.subheader("2. Hasil Deteksi & Analisis Mendalam")
        
        if 'analyze_btn' in locals() and analyze_btn:
            if not api_key:
                st.error("⚠️ **API Key Belum Diisi!** Silakan masukkan Gemini API Key pada sidebar sebelah kiri atau simpan di Streamlit Secrets.")
            elif input_image is None:
                st.warning("⚠️ **Gambar Belum Ada!** Ambil foto atau unggah gambar e-waste terlebih dahulu.")
            else:
                with st.spinner("🧠 Menganalisis gambar dengan Vision AI Engine..."):
                    data, active_model, err = analyze_ewaste_master(input_image, api_key)
                    
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state.detection_history.append({
                            "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "nama": data.get("nama_objek", "Tidak diketahui"),
                            "kategori": data.get("kategori_un", "Umum"),
                            "bahaya": data.get("tingkat_bahaya", "Sedang"),
                            "daur_ulang": data.get("dapat_didaur_ulang_persen", 0),
                            "model": active_model
                        })
                        
                        st.success(f"✅ **Berhasil Dianalisis** (Model Digunakan: `{active_model}`)")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Perangkat Terdeteksi", data.get("nama_objek", "-"))
                        m2.metric("Tingkat Bahaya", data.get("tingkat_bahaya", "-"), delta=f"Skor {data.get('skor_bahaya', 0)}/10", delta_color="inverse")
                        m3.metric("Potensi Daur Ulang", f"{data.get('dapat_didaur_ulang_persen', 0)}%")
                        
                        st.markdown("---")
                        st.markdown(f"**📂 Kategori UN E-Waste:** `{data.get('kategori_un', '-')}`")
                        st.markdown(f"**📝 Deskripsi Objek:** {data.get('deskripsi', '-')}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("🚨 **Bahan / Zat Berbahaya:**")
                            for bahan in data.get("bahan_berbahaya", []):
                                st.write(f"- {bahan}")
                                
                        with col_b:
                            st.markdown("💎 **Potensi Logam Mulia:**")
                            lm = data.get("potensi_logam_mulia", {})
                            for k, v in lm.items():
                                st.write(f"- **{k}:** {v}")
                                
                        st.markdown("---")
                        st.markdown("🛠️ **Instruksi Penanganan & Daur Ulang Aman:**")
                        for idx, step in enumerate(data.get("instruksi_penanganan", []), 1):
                            st.write(f"**{idx}.** {step}")

# --- TAB 2: DASHBOARD & RIWAYAT ---
with tab2:
    st.subheader("📊 Rekapitulasi Deteksi E-Waste")
    if len(st.session_state.detection_history) > 0:
        df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df, names="kategori", title="Distribusi Kategori UN E-Waste", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_bar = px.bar(df, x="nama", y="daur_ulang", color="bahaya", title="Persentase Daur Ulang per Objek")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Belum ada riwayat deteksi pada sesi ini. Lakukan deteksi di Tab 1 untuk melihat dashboard.")
