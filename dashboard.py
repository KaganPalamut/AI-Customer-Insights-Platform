import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import requests
from bs4 import BeautifulSoup

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Strateji Merkezi v2.1", layout="wide")

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Model dosyaları bulunamadı: {e}")
        return None, None

model, vectorizer = load_models()

# --- YARDIMCI FONKSİYONLAR ---
def scrape_comments(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        comments = [p.text.strip() for p in soup.find_all("p") if len(p.text.strip()) > 15]
        return pd.DataFrame({"Yorum": comments})
    except Exception as e:
        st.error(f"Veri çekilirken hata oluştu: {e}")
        return None

def detect_detailed_category(text):
    text = str(text).lower()
    logic_map = {
        "Kargo Gecikmesi": ["geç geldi", "gecikti", "ulaşmadı", "kargo", "teslimat", "bekliyorum"],
        "Hasarlı Ürün": ["kırık", "ezik", "hasarlı", "yırtık", "parçalanmış", "bozuk", "defolu"],
        "Fiyat Şikayeti": ["pahalı", "değmez", "fiyatı", "zam", "maliyet"],
        "Müşteri İlgisizliği": ["cevap vermiyor", "muhatap", "temsilci", "ilgisiz", "destek"],
        "Kalite Sorunu": ["kalitesiz", "beklediğim", "kötü", "çöp", "berbat"],
        "İade Problemi": ["iade", "paramı", "süreç", "geri gönderdim"]
    }
    for cat, keywords in logic_map.items():
        if any(word in text for word in keywords):
            return cat
    return "Diğer/Genel"

# --- ARAYÜZ (SIDEBAR) ---
st.sidebar.title("🛠️ Veri Giriş Merkezi")
uploaded_file = st.sidebar.file_uploader("CSV Dosyası Yükle", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Web'den Veri Çek")
url_input = st.sidebar.text_input("URL girin:", value="https://tr.wikipedia.org/wiki/Türkiye")
scrape_btn = st.sidebar.button("Verileri Kazı")

if 'df' not in st.session_state:
    st.session_state['df'] = None

# Veri Yükleme Mantığı
if uploaded_file:
    try:
        st.session_state['df'] = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        uploaded_file.seek(0)
        st.session_state['df'] = pd.read_csv(uploaded_file, encoding='latin-1')
elif scrape_btn and url_input:
    scraped_data = scrape_comments(url_input)
    if scraped_data is not None:
        st.session_state['df'] = scraped_data

df = st.session_state['df']

# --- ANA PANEL ---
st.title("🚀 AI Müşteri Deneyimi & Strateji Paneli")

if df is not None and not df.empty:
    try:
        # Analiz Süreci
        X_transformed = vectorizer.transform(df['Yorum'])
        df['Duygu'] = model.predict(X_transformed)
        df['Detaylı Kategori'] = df['Yorum'].apply(detect_detailed_category)

        # 1. Üst Metrikler
        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam Veri", len(df))
        
        neg_sayisi = len(df[df['Duygu'] == 'Negatif'])
        m2.metric("Negatif Tahmini", neg_sayisi)
        
        # Sorun tespiti
        gercek_sorunlar_df = df[df['Detaylı Kategori'] != "Diğer/Genel"]
        top_issue = gercek_sorunlar_df['Detaylı Kategori'].mode()[0] if not gercek_sorunlar_df.empty else "Sorun Yok"
        m3.metric("En Büyük Sorun", top_issue)

        # 2. Grafikler
        g1, g2 = st.columns(2)
        with g1:
            fig_pie = px.pie(df, names='Duygu', title="Duygu Dağılımı", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        with g2:
            kat_sayi = df['Detaylı Kategori'].value_counts().reset_index()
            kat_sayi.columns = ['Kategori', 'Adet']
            fig_bar = px.bar(kat_sayi, x='Kategori', y='Adet', title="Konu Dağılımı", color='Kategori')
            st.plotly_chart(fig_bar, use_container_width=True)

        # 3. AI Strateji Motoru
        st.markdown("---")
        st.header("💡 AI Strateji Motoru")
        
        if top_issue != "Sorun Yok":
            st.warning(f"Yoğun Şikayet Odağı: **{top_issue}**")
            advice_logic = {
                "Kargo Gecikmesi": "🚚 **Eylem:** Lojistik partneriyle performans görüşmesi yapın.",
                "Hasarlı Ürün": "📦 **Eylem:** Paketleme kontrolünü artırın ve kargo tazmin süreci başlatın.",
                "Fiyat Şikayeti": "🏷️ **Eylem:** Rakip analizi yapın, fiyat-performans vurgulu kampanya yapın.",
                "Müşteri İlgisizliği": "📞 **Eylem:** Destek ekibi kapasitesini artırın.",
                "Kalite Sorunu": "🔍 **Eylem:** Tedarik denetimini sıkılaştırın.",
                "İade Problemi": "🔄 **Eylem:** İade sürecini hızlandırın."
            }
            st.info(advice_logic.get(top_issue, "Genel iyileştirme süreci başlatın."))
        else:
            st.success("Harika! Belirgin bir şikayet odağı bulunamadı.")

        with st.expander("Tüm Verileri Gör"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"Analiz sırasında hata oluştu: {e}")
else:
    st.info("📊 Başlamak için veri yükleyin veya bir URL girin.")