import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Sayfa Ayarları
st.set_page_config(page_title="AI Business Intelligence", layout="wide")
st.title("🚀 AI Müşteri Deneyimi & Strateji Merkezi")
st.markdown("---")

# Modelleri Yükle
@st.cache_resource # Modeli bir kez yükle, hızı artır
def load_models():
    with open("sentiment_model.pkl", "rb") as f: model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_models()
    
    # Dosya Yükleme Alanı
    st.sidebar.header("📊 Veri Kaynağı")
    uploaded_file = st.sidebar.file_uploader("Müşteri Yorumları (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # 1. AI Analiz Süreci
        df['sentiment'] = model.predict(vectorizer.transform(df['text'].astype(str)))
        
        # Konu Belirleme Mantığı
        def get_topic(text):
            t = str(text).lower()
            if any(k in t for k in ["kargo", "teslim", "hız", "geç"]): return "Lojistik"
            if any(k in t for k in ["fiyat", "pahalı", "para", "indirim"]): return "Ekonomi"
            if any(k in t for k in ["kalite", "bozuk", "sağlam", "malzeme"]): return "Ürün"
            return "Genel Şikayet"
        
        df['topic'] = df['text'].apply(get_topic)

        # 2. Üst Özet Kartları
        c1, c2, c3 = st.columns(3)
        total_comments = len(df)
        neg_count = len(df[df['sentiment'] == 'negatif'])
        pos_ratio = (len(df[df['sentiment'] == 'pozitif']) / total_comments) * 100

        c1.metric("Toplam Yorum", total_comments)
        c2.metric("Negatif Yorum", neg_count, delta="-Kritik", delta_color="inverse")
        c3.metric("Memnuniyet Skoru", f"%{pos_ratio:.1f}")

        # 3. Görsel Analiz
        st.markdown("### 📈 Veri Görselleştirme")
        col_left, col_right = st.columns(2)
        
        with col_left:
            fig_pie = px.pie(df, names='sentiment', title="Genel Müşteri Duygusu",
                             color_discrete_map={'pozitif':'#2ecc71','negatif':'#e74c3c','notr':'#95a5a6'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            neg_topics = df[df['sentiment'] == 'negatif']['topic'].value_counts().reset_index()
            fig_bar = px.bar(neg_topics, x='topic', y='count', title="Sorunların Odak Noktası",
                             labels={'topic':'Konu', 'count':'Şikayet Sayısı'}, color='count')
            st.plotly_chart(fig_bar, use_container_width=True)

        # 4. 🧠 AI STRATEJİ VE TAVSİYE MOTORU
        st.markdown("---")
        st.subheader("💡 Yapay Zeka Stratejik Tavsiyeleri")
        
        with st.expander("Detaylı Analiz Raporunu Gör", expanded=True):
            # En çok şikayet edilen konuyu bul
            if not neg_topics.empty:
                top_issue = neg_topics.iloc[0]['topic']
                
                st.warning(f"🚨 **Kritik Tespit:** En büyük sorun alanı **{top_issue}** olarak görünüyor.")
                
                # Dinamik Tavsiye Üretimi
                if top_issue == "Lojistik":
                    st.write("- Kargo firması ile olan anlaşmalarınızı gözden geçirin.")
                    st.write("- Teslimat süresi vaatlerinizi güncelleyin.")
                elif top_issue == "Ekonomi":
                    st.write("- Fiyat/Performans algısını güçlendirmek için kampanya kurgulayın.")
                    st.write("- Sadakat programları (puan/indirim) başlatmayı düşünün.")
                elif top_issue == "Ürün":
                    st.write("- Üretim bandında kalite kontrol denetimlerini artırın.")
                    st.write("- İade süreçlerini kolaylaştırarak güven tazeleyin.")
            else:
                st.success("Tebrikler! Belirgin bir şikayet odağı bulunamadı.")

except Exception as e:
    st.info("Lütfen bir CSV dosyası yükleyerek analizi başlatın.")