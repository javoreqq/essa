import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Biblioteki do Machine Learningu (dodane na Twoją prośbę)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Konfiguracja strony
st.set_page_config(page_title="Wino AI - Analiza i Predykcja", layout="wide")

# --- FUNKCJA ŁADUJĄCA DANE ---
@st.cache_data
def load_data():
    try:
        df_quality = pd.read_csv('winequality-red.csv')
        df_pairing = pd.read_csv('wine_food_pairings.csv')
        return df_quality, df_pairing
    except FileNotFoundError:
        return None, None

# Wczytanie danych
df_red, df_pair = load_data()

# --- GŁÓWNA LOGIKA APLIKACJI ---

if df_red is not None and df_pair is not None:
    
    st.title("🍷 Wino AI: Analiza, Sommelier i Predykcja")
    st.markdown("Kompletne narzędzie dla enologów i smakoszy.")

    # TERAZ MAMY 3 ZAKŁADKI (doszła Predykcja AI)
    tab1, tab2, tab3 = st.tabs(["📊 Analiza Danych", "🍽️ Wirtualny Sommelier", "🤖 Predykcja Jakości (ML)"])

    # --- ZAKŁADKA 1: ANALIZA ---
    with tab1:
        st.header("Eksploracja danych chemicznych")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Statystyki opisowe:")
            st.dataframe(df_red.describe().T[['mean', 'std', 'min', 'max']])
        
        with col2:
            st.write("Korelacja parametrów z jakością:")
            # Obliczamy korelację tylko dla kolumny 'quality'
            corr_matrix = df_red.corr()[['quality']].sort_values(by='quality', ascending=False)
            fig_corr, ax_corr = plt.subplots(figsize=(6, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr, vmin=-1, vmax=1)
            st.pyplot(fig_corr)

    # --- ZAKŁADKA 2: SOMMELIER ---
    with tab2:
        st.header("System rekomendacji kulinarnych")
        
        # Prosty interfejs wyboru
        option = st.selectbox("Wybierz typ wina:", sorted(df_pair['wine_type'].unique()))
        
        st.write(f"Najlepsze potrawy do: **{option}**")
        
        # Filtrowanie najlepszych połączeń (ocena >= 4)
        best_pairs = df_pair[
            (df_pair['wine_type'] == option) & 
            (df_pair['pairing_quality'] >= 4)
        ].sort_values(by='pairing_quality', ascending=False)
        
        if not best_pairs.empty:
            st.dataframe(best_pairs[['food_item', 'cuisine', 'pairing_quality', 'description']], hide_index=True)
        else:
            st.info("Brak wybitnych rekomendacji w bazie dla tego szczepu.")

    # --- ZAKŁADKA 3: MODEL ML (NOWOŚĆ!) ---
    with tab3:
        st.header("Przewidywanie jakości wina (Random Forest)")
        st.markdown("Model uczy się na bazie danych historycznych i ocenia Twoje wino.")

        # 1. Przygotowanie danych
        X = df_red.drop('quality', axis=1)
        y = df_red['quality']
        
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Trenowanie modelu
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # 3. Ewaluacja modelu (wyświetlamy metryki)
        y_pred = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Dokładność modelu (R2 Score)", f"{r2:.2f}")
        col_m2.metric("Średni błąd (MAE)", f"{mae:.2f}")
        
        st.divider()
        st.subheader("Sprawdź swoje wino!")
        
        # 4. Interfejs do wprowadzania danych przez użytkownika
        # Tworzymy slidery dla 4 najważniejszych cech (żeby nie zaśmiecać ekranu)
        col_inp1, col_inp2 = st.columns(2)
        
        with col_inp1:
            alcohol = st.slider("Alkohol (%)", 8.0, 15.0, 10.5)
            sulphates = st.slider("Siarczany (Sulphates)", 0.3, 2.0, 0.65)
            
        with col_inp2:
            volatile_acidity = st.slider("Kwasowość lotna (Volatile Acidity)", 0.1, 1.6, 0.5)
            citric_acid = st.slider("Kwas cytrynowy", 0.0, 1.0, 0.25)
            
        # Pobieramy średnie wartości dla pozostałych cech (żeby model miał komplet danych)
        input_data = pd.DataFrame([X.mean().values], columns=X.columns)
        
        # Nadpisujemy wartościami wybranymi przez użytkownika
        input_data['alcohol'] = alcohol
        input_data['sulphates'] = sulphates
        input_data['volatile acidity'] = volatile_acidity
        input_data['citric acid'] = citric_acid
        
        # 5. Przycisk predykcji
        if st.button("Oceń jakość wina"):
            prediction = rf_model.predict(input_data)[0]
            
            st.success(f"Przewidywana jakość wina: {prediction:.2f} / 10")
            
            if prediction > 6.5:
                st.balloons()
                st.markdown("🌟 **To może być wybitne wino!**")
            elif prediction < 5.0:
                st.markdown("💀 **Raczej słabej jakości...**")
            else:
                st.markdown("🍷 **Solidne, stołowe wino.**")

# Obsługa braku plików (z poprzedniego kroku)
else:
    st.error("Brak plików danych!")
    st.info("Wgraj pliki 'winequality-red.csv' oraz 'wine_food_pairings.csv' poniżej:")
    
    up1 = st.file_uploader("Plik jakości (winequality-red)", type='csv')
    up2 = st.file_uploader("Plik parowania (wine_food_pairings)", type='csv')
    
    if up1 and up2:
        df_red = pd.read_csv(up1)
        df_pair = pd.read_csv(up2)
        st.rerun()
