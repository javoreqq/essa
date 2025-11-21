import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Nowe biblioteki do interaktywnych wykresów
import plotly.express as px
import plotly.graph_objects as go

# Biblioteki ML (Dodano XGBoost)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Konfiguracja strony
st.set_page_config(page_title="Wino Expert Pro - XGBoost & Plotly", layout="wide")

# Stylizacja wykresów Seaborn (dla statycznych wykresów)
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
    
    st.title("🍇 Wino Expert Pro: XGBoost & Plotly")
    st.markdown("Zaawansowana analityka, interaktywne wizualizacje 3D oraz porównanie modeli AI.")

    # 4 Zakładki
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Eksploracja (EDA)", 
        "🔬 Zaawansowana Wizualizacja (Plotly)",
        "🤖 Modele AI (XGBoost vs RF)",
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
            st.subheader("Rozkłady zmiennych")
            feature_to_plot = st.selectbox("Wybierz cechę:", df_red.columns)
            
            # Używamy Plotly zamiast Matplotlib dla lepszego efektu
            fig_hist = px.histogram(df_red, x=feature_to_plot, nbins=30, title=f"Histogram: {feature_to_plot}", color_discrete_sequence=['darkred'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            st.subheader("Statystyki szczegółowe")
            st.dataframe(df_red.describe().T.style.background_gradient(cmap="Reds"))

    # --- ZAKŁADKA 2: ZAAWANSOWANA WIZUALIZACJA (3 METODY PLOTLY) ---
    with tab2:
        st.header("Interaktywna Analiza Danych")
        st.markdown("Trzy metody wizualizacji wielowymiarowej.")
        
        # METODA 1: 3D SCATTER PLOT
        st.subheader("1. Wykres 3D (Trzy wymiary jakości)")
        col_3d_1, col_3d_2, col_3d_3 = st.columns(3)
        x_axis = col_3d_1.selectbox("Oś X", df_red.columns, index=10) # alcohol
        y_axis = col_3d_2.selectbox("Oś Y", df_red.columns, index=1)  # volatile acidity
        z_axis = col_3d_3.selectbox("Oś Z", df_red.columns, index=9)  # sulphates
        
        fig_3d = px.scatter_3d(
            df_red, x=x_axis, y=y_axis, z=z_axis,
            color='quality', opacity=0.7,
            color_continuous_scale='Viridis',
            title=f"Relacja 3D"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.divider()
        
        col_v1, col_v2 = st.columns(2)
        
        # METODA 2: VIOLIN PLOT
        with col_v1:
            st.subheader("2. Wykres Skrzypcowy (Violin Plot)")
            st.caption("Analiza gęstości rozkładu dla poszczególnych ocen.")
            y_violin = st.selectbox("Cecha do analizy:", df_red.columns[:-1], index=10)
            
            fig_violin = px.violin(
                df_red, y=y_violin, x="quality", 
                box=True, points="all", 
                color="quality",
                title=f"Rozkład {y_violin} wg Jakości"
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
        # METODA 3: PARALLEL COORDINATES
        with col_v2:
            st.subheader("3. Współrzędne Równoległe")
            st.caption("Śledzenie profilu chemicznego najlepszych win.")
            
            # Wybieramy kluczowe cechy dla czytelności
            cols_parallel = ['alcohol', 'sulphates', 'volatile acidity', 'pH', 'quality']
            
            fig_par = px.parallel_coordinates(
                df_red[cols_parallel], 
                color="quality",
                labels={"alcohol": "Alkohol", "sulphates": "Siarczany", "quality": "Jakość"},
                color_continuous_scale=px.colors.diverging.Tealrose,
                color_continuous_midpoint=5.5
            )
            st.plotly_chart(fig_par, use_container_width=True)

    # --- ZAKŁADKA 3: MODELE AI (XGBOOST & RF) ---
    with tab3:
        st.header("Predykcja Jakości Wina")
        
        col_sett1, col_sett2 = st.columns([1, 3])
        
        with col_sett1:
            st.subheader("Konfiguracja Modelu")
            # Wybór modelu
            model_type = st.radio("Wybierz algorytm:", ["XGBoost (Gradient Boosting)", "Random Forest"])
            test_size = st.slider("Zbiór testowy (%)", 10, 40, 20) / 100.0
            
        with col_sett2:
            # Trenowanie modelu "w locie"
            X = df_red.drop('quality', axis=1)
            y = df_red['quality']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                # Konfiguracja XGBoost
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Wyniki
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            st.write(f"### Wyniki dla: {model_type}")
            m1, m2, m3 = st.columns(3)
            m1.metric("R2 Score", f"{r2:.3f}")
            m2.metric("MAE (Błąd)", f"{mae:.3f}")
            m3.metric("RMSE", f"{rmse:.3f}")
            
            # Feature Importance (Wykres słupkowy Plotly)
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Cecha': X.columns, 'Waga': importances}).sort_values(by='Waga', ascending=True)
            
            fig_imp = px.bar(feat_df, x='Waga', y='Cecha', orientation='h', title="Wpływ cech chemicznych na jakość", color='Waga')
            st.plotly_chart(fig_imp, use_container_width=True, height=300)

        st.divider()
        st.subheader("🧪 Symulator Sommeliera AI")
        
        # Inputy użytkownika
        c1, c2, c3, c4 = st.columns(4)
        val_alc = c1.slider("Alkohol (%)", 8.0, 15.0, 10.5)
        val_sul = c2.slider("Siarczany", 0.3, 2.0, 0.65)
        val_vol = c3.slider("Kwasowość lotna", 0.1, 1.6, 0.5)
        val_cit = c4.slider("Kwas cytrynowy", 0.0, 1.0, 0.25)
        
        # Przygotowanie danych do predykcji
        input_vector = pd.DataFrame([X.mean().values], columns=X.columns)
        input_vector['alcohol'] = val_alc
        input_vector['sulphates'] = val_sul
        input_vector['volatile acidity'] = val_vol
        input_vector['citric acid'] = val_cit
        
        if st.button("Oceń wino"):
            pred_val = model.predict(input_vector)[0]
            
            # Wizualizacja wyniku na liczniku (Gauge Chart)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Ocena ({model_type})"},
                gauge = {
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 7], 'color': "gray"},
                        {'range': [7, 10], 'color': "gold"}],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=False)

    # --- ZAKŁADKA 4: SOMMELIER ---
    with tab4:
        st.header("Baza wiedzy o parowaniu (Wine Pairing)")
        
        search_term = st.text_input("Wpisz nazwę potrawy lub wina (np. 'lamb', 'Merlot'):", "")
        
        if search_term:
            mask = df_pair.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            results = df_pair[mask]
            
            if not results.empty:
                st.success(f"Znaleziono {len(results)} pasujących rekordów.")
                st.dataframe(results[['wine_type', 'food_item', 'pairing_quality', 'description']], hide_index=True)
            else:
                st.warning("Nie znaleziono pasujących wyników.")
        else:
            st.info("Zacznij pisać powyżej, aby przeszukać bazę sommeliera.")
            st.dataframe(df_pair.head(5))

else:
    st.error("⚠️ Nie znaleziono danych.")
    st.info("Wgraj pliki CSV, aby uruchomić dashboard.")
    
    u1 = st.file_uploader("winequality-red.csv", type='csv')
    u2 = st.file_uploader("wine_food_pairings.csv", type='csv')
    
    if u1 and u2:
        df_red = pd.read_csv(u1)
        df_pair = pd.read_csv(u2)
        st.success("Dane wczytane! Odświeżam...")
        st.rerun()
