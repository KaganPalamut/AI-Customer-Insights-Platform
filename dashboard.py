
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import requests
from bs4 import BeautifulSoup

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Strateji Merkezi", layout="wide")

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# --- YARDIMCI FONKSİYONLAR ---
def scrape_comments(url):
    """Verilen URL'deki paragrafları çekerek veri çerçevesine dönüştürür."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        # 10 karakterden uzun metinleri topluyoruz
        comments = [p.text.strip() for p in soup.find_all("p") if len(p.text.strip()) > 10]
        return pd.DataFrame({"Yorum": comments})
    except Exception as e:
        st.error(f"Veri çekilirken hata oluştu: {e}")
        return None

def detect_category(text):
    """Metin içindeki anahtar kelimelere göre kategori belirler."""
    text = text.lower()
    categories = {
        "Lojistik/Kargo": ["kargo", "teslimat", "gecikti", "gelmedi", "paketleme", "kurye", "lojistik"],
        "Ürün Kalitesi": ["bozuk", "kırık", "kalitesiz", "yırtık", "çalışmıyor", "defolu", "malzeme"],
        "Fiyat/Ekonomi": ["pahalı", "zam", "fiyat", "iade", "ücret", "indirim", "ekonomi"],
        "Müşteri Hizmetleri": ["destek", "temsilci", "muhatap", "cevap", "aramadı", "ilgisiz"]
    }
    for cat, keywords in categories.items():
        if any(word in text for word in keywords):
            return cat
    return "Diğer"

# --- ARAYÜZ (SIDEBAR) ---
st.sidebar.title("🛠️ Veri Giriş Merkezi")

# Seçenek 1: Dosya Yükleme
uploaded_file = st.sidebar.file_uploader("CSV Dosyası Yükle", type=["csv"])

st.sidebar.markdown("---")

# Seçenek 2: Web Scraping
st.sidebar.subheader("🌐 Web'den Veri Çek")
url_input = st.sidebar.text_input("Yorum çekilecek URL:", value="https://tr.wikipedia.org/wiki/Türkiye")
scrape_btn = st.sidebar.button("Verileri Kazı")

# Veri Kaynağı Kontrolü
if 'df' not in st.session_state:
    st.session_state['df'] = None

if uploaded_file:
    st.session_state['df'] = pd.read_csv(uploaded_file)
elif scrape_btn and url_input:
    with st.spinner("İnternetten veriler çekiliyor..."):
        scraped_data = scrape_comments(url_input)
        if scraped_data is not None:
            st.session_state['df'] = scraped_data
            st.sidebar.success(f"{len(scraped_data)} satır veri bulundu!")

df = st.session_state['df']

# --- ANA PANEL ---
st.title("🚀 AI Müşteri Deneyimi & Strateji Paneli")

if df is not None and "Yorum" in df.columns:
    # 1. Analizleri Yap
    df['Duygu'] = model.predict(vectorizer.transform(df['Yorum']))
    df['Kategori'] = df['Yorum'].apply(detect_category)

    # 2. Metrikler
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Veri", len(df))
    col2.metric("Negatif Sayısı", len(df[df['Duygu'] == 'Negatif']))
    col3.metric("Kritik Odak", df['Kategori'].mode()[0] if not df['Kategori'].empty else "Belirlenemedi")

    # 3. Grafikler
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(df, names='Duygu', title="Duygu Analizi Dağılımı", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c2:
        # Hatalı olan kısım burasıydı, düzeltildi:
        kategori_ozet = df['Kategori'].value_counts().reset_index()
        kategori_ozet.columns = ['Kategori', 'Adet']
        fig_bar = px.bar(kategori_ozet, x='Kategori', y='Adet', title="Kategori Bazlı Dağılım", color='Kategori')
        st.plotly_chart(fig_bar, use_container_width=True)

    # 4. AI Stratejik Tavsiyeler
    st.markdown("---")
    st.header("💡 AI Stratejik Tavsiye Motoru")
    
    neg_df = df[df['Duygu'] == 'Negatif']
    if not neg_df.empty:
        top_issue = neg_df['Kategori'].value_counts().idxmax()
        
        advice_map = {
            "Lojistik/Kargo": "⚠️ Lojistik süreçlerde aksama tespit edildi. Kargo firması ile görüşülmeli.",
            "Ürün Kalitesi": "🛠️ Ürün kalitesine yönelik spesifik şikayetler var. Ar-Ge birimi bilgilendirilmeli.",
            "Fiyat/Ekonomi": "💰 Fiyat algısı negatif eğilimde. Kampanya veya indirim planlanabilir.",
            "Müşteri Hizmetleri": "📞 Müşteri temsilcisi yanıt süreleri veya tavırları iyileştirilmeli.",
            "Diğer": "🔍 Verilerde genel bir memnuniyetsizlik var, detaylı anket yapılmalı."
        }
        st.info(f"**Ana Sorun Kaynağı:** {top_issue}\n\n**Önerilen Strateji:** {advice_map.get(top_issue)}")
    else:
        st.success("Analiz edilen verilerde kritik bir negatif duruma rastlanmadı.")

    # 5. Tablo
    with st.expander("Veri Detaylarını Gör"):
        st.dataframe(df)

else:
    st.info("Analiz başlamak için soldan dosya yükleyin veya bir URL girip 'Verileri Kazı' butonuna basın.")