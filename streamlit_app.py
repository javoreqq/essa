import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Biblioteki ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Konfiguracja strony
st.set_page_config(page_title="Wino Expert - Analytics Dashboard", layout="wide")

# Stylizacja wykresów Seaborn
sns.set_theme(style="whitegrid")

# --- FUNKCJA ŁADUJĄCA DANE ---
@st.cache_data
def load_data():
    try:
        df_quality = pd.read_csv('winequality-red.csv')
        df_pairing = pd.read_csv('wine_food_pairings.csv')
        return df_quality, df_pairing
    except FileNotFoundError:
        return None, None

df_red, df_pair = load_data()

# --- GŁÓWNA APLIKACJA ---

if df_red is not None and df_pair is not None:
    
    st.title("🍷 Wino Expert: Analityka, AI i Sommelier")
    st.markdown("Zaawansowane narzędzie do wizualizacji danych winiarskich i predykcji jakości.")

    # 4 Zakładki zamiast 3
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Eksploracja Danych (EDA)", 
        "🔬 Zaawansowana Wizualizacja",
        "🤖 Model AI & Ważność Cech",
        "🍽️ Sommelier"
    ])

    # --- ZAKŁADKA 1: EKSPLORACJA (EDA) ---
    with tab1:
        st.header("Podstawowa analiza statystyczna")
        
        # Szybkie KPI
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Średnia Jakość", f"{df_red['quality'].mean():.2f}")
        kpi2.metric("Średni Alkohol", f"{df_red['alcohol'].mean():.1f}%")
        kpi3.metric("Liczba próbek", df_red.shape[0])
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rozkłady zmiennych (Histogramy)")
            feature_to_plot = st.selectbox("Wybierz cechę do analizy:", df_red.columns)
            
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(data=df_red, x=feature_to_plot, kde=True, color="darkred", ax=ax_hist)
            ax_hist.set_title(f"Rozkład: {feature_to_plot}")
            st.pyplot(fig_hist)
            
        with col2:
            st.subheader("Statystyki szczegółowe")
            st.dataframe(df_red.describe().T.style.background_gradient(cmap="Reds"))

    # --- ZAKŁADKA 2: ZAAWANSOWANA WIZUALIZACJA ---
    with tab2:
        st.header("Kreator Wykresów")
        st.markdown("Szukaj zależności pomiędzy dowolnymi parametrami.")
        
        c1, c2, c3 = st.columns(3)
        x_axis = c1.selectbox("Oś X", df_red.columns, index=10) # Domyślnie alcohol
        y_axis = c2.selectbox("Oś Y", df_red.columns, index=1)  # Domyślnie volatile acidity
        color_by = c3.selectbox("Kolorowanie (Hue)", [None, 'quality'], index=1)
        
        # Wykres punktowy (Scatter Plot)
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        
        if color_by == 'quality':
            # Traktujemy jakość jako kategorię dla lepszych kolorów
            sns.scatterplot(data=df_red, x=x_axis, y=y_axis, hue='quality', palette='viridis', ax=ax_scatter, s=60, alpha=0.7)
        else:
            sns.scatterplot(data=df_red, x=x_axis, y=y_axis, color='darkred', ax=ax_scatter, s=60, alpha=0.7)
            
        ax_scatter.set_title(f"Zależność: {x_axis} vs {y_axis}")
        st.pyplot(fig_scatter)
        
        st.divider()
        st.subheader("Filtrowanie danych")
        
        # Prosty filtr
        min_quality = st.slider("Pokaż wina o jakości co najmniej:", 3, 8, 5)
        filtered_df = df_red[df_red['quality'] >= min_quality]
        
        st.write(f"Znaleziono **{filtered_df.shape[0]}** win spełniających kryteria.")
        with st.expander("Pokaż przefiltrowaną tabelę"):
            st.dataframe(filtered_df)
            
            # Opcja pobrania danych
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Pobierz przefiltrowane dane (CSV)",
                csv,
                "filtered_wine.csv",
                "text/csv",
                key='download-csv'
            )

    # --- ZAKŁADKA 3: MODEL AI & FEATURE IMPORTANCE ---
    with tab3:
        st.header("Random Forest Regressor")
        
        # Trenowanie modelu (cache'owane, aby nie liczyć przy każdym kliknięciu)
        @st.cache_resource
        def train_model(data):
            X = data.drop('quality', axis=1)
            y = data['quality']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            return model, r2, mae, X.columns

        model, r2, mae, feature_names = train_model(df_red)

        # Metryki
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("R2 Score (Dokładność)", f"{r2:.2%}")
        col_m2.metric("MAE (Średni błąd)", f"{mae:.2f}")
        
        st.divider()
        
        # --- WAŻNOŚĆ CECH (NOWOŚĆ) ---
        st.subheader("🔍 Co najbardziej wpływa na jakość wina?")
        st.markdown("Wykres pokazuje, które parametry chemiczne były najważniejsze dla modelu przy ocenie wina.")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Tworzenie DataFrame do wykresu
        feat_df = pd.DataFrame({
            'Cecha': [feature_names[i] for i in indices],
            'Waga': importances[indices]
        })
        
        fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Waga', y='Cecha', data=feat_df, palette='magma', ax=ax_feat)
        ax_feat.set_title("Feature Importance (Ważność Cech)")
        st.pyplot(fig_feat)
        
        st.info("💡 **Interpretacja:** Cecha na samej górze (zazwyczaj 'Alcohol' lub 'Sulphates') ma największy wpływ na to, czy wino dostanie wysoką ocenę.")

        # Interaktywna predykcja (znana z poprzedniej wersji)
        st.divider()
        st.subheader("Symulator jakości")
        
        c_input1, c_input2, c_input3, c_input4 = st.columns(4)
        val_alc = c_input1.slider("Alcohol", 8.0, 15.0, 10.0)
        val_sul = c_input2.slider("Sulphates", 0.3, 2.0, 0.6)
        val_vol = c_input3.slider("Volatile Acidity", 0.1, 1.6, 0.5)
        val_cit = c_input4.slider("Citric Acid", 0.0, 1.0, 0.25)
        
        # Tworzymy wektor wejściowy ze średnimi wartościami
        input_vector = pd.DataFrame([df_red.drop('quality', axis=1).mean().values], columns=feature_names)
        # Podmieniamy to co użytkownik zmienił
        input_vector['alcohol'] = val_alc
        input_vector['sulphates'] = val_sul
        input_vector['volatile acidity'] = val_vol
        input_vector['citric acid'] = val_cit
        
        if st.button("Oblicz prognozowaną ocenę"):
            pred_val = model.predict(input_vector)[0]
            st.metric("Przewidywana Jakość", f"{pred_val:.2f} / 10")

    # --- ZAKŁADKA 4: SOMMELIER ---
    with tab4:
        st.header("Baza wiedzy o parowaniu (Wine Pairing)")
        
        # Wyszukiwarka
        search_term = st.text_input("Wpisz nazwę potrawy lub wina (np. 'lamb', 'Merlot'):", "")
        
        if search_term:
            # Filtrowanie po wielu kolumnach
            mask = df_pair.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            results = df_pair[mask]
            
            if not results.empty:
                st.success(f"Znaleziono {len(results)} pasujących rekordów.")
                st.dataframe(results[['wine_type', 'food_item', 'pairing_quality', 'description']], hide_index=True)
            else:
                st.warning("Nie znaleziono pasujących wyników.")
        else:
            st.info("Zacznij pisać powyżej, aby przeszukać bazę sommeliera.")
            st.write("Przykładowe dane:")
            st.dataframe(df_pair.head(5))

else:
    # --- EKRAN STARTOWY (JEŚLI BRAK PLIKÓW) ---
    st.error("⚠️ Nie znaleziono danych.")
    st.info("Wgraj pliki CSV, aby uruchomić dashboard.")
    
    u1 = st.file_uploader("winequality-red.csv", type='csv')
    u2 = st.file_uploader("wine_food_pairings.csv", type='csv')
    
    if u1 and u2:
        df_red = pd.read_csv(u1)
        df_pair = pd.read_csv(u2)
        st.success("Dane wczytane! Odświeżam...")
        st.rerun()
