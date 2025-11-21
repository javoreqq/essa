import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Biblioteki ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Konfiguracja strony
st.set_page_config(page_title="Wino AI 2.0 - XGBoost & Plotly", layout="wide")

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
    
    st.title("🍇 Wino AI 2.0: XGBoost & Plotly")
    st.markdown("Zaawansowana analityka z interaktywnymi wykresami i modelowaniem Gradient Boosting.")

    # Zakładki
    tab1, tab2, tab3 = st.tabs([
        "📈 Wizualizacja 3D i Interakcja", 
        "🤖 Modele ML (XGBoost vs RF)",
        "🍽️ Sommelier"
    ])

    # --- ZAKŁADKA 1: WIZUALIZACJE PLOTLY (3 METODY) ---
    with tab1:
        st.header("Zaawansowana Wizualizacja Danych")
        
        # METODA 1: Wykres 3D (3D Scatter Plot)
        st.subheader("1. Analiza wielowymiarowa (3D Scatter)")
        st.caption("Obracaj wykresem myszką, aby zobaczyć klastry jakości.")
        
        col_3d_1, col_3d_2, col_3d_3 = st.columns(3)
        x_axis = col_3d_1.selectbox("Oś X", df_red.columns, index=10) # alcohol
        y_axis = col_3d_2.selectbox("Oś Y", df_red.columns, index=1)  # volatile acidity
        z_axis = col_3d_3.selectbox("Oś Z", df_red.columns, index=9)  # sulphates
        
        fig_3d = px.scatter_3d(
            df_red, x=x_axis, y=y_axis, z=z_axis,
            color='quality', opacity=0.7,
            color_continuous_scale='Viridis',
            title=f"Relacja 3D: {x_axis} vs {y_axis} vs {z_axis}"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.divider()

        col_v1, col_v2 = st.columns(2)

        # METODA 2: Wykres Skrzypcowy (Violin Plot)
        with col_v1:
            st.subheader("2. Rozkład gęstości (Violin Plot)")
            st.caption("Pokazuje nie tylko średnią, ale też kształt rozkładu danych.")
            
            y_violin = st.selectbox("Wybierz parametr do analizy rozkładu:", df_red.columns[:-1], index=1)
            
            fig_violin = px.violin(
                df_red, y=y_violin, x="quality", 
                box=True, points="all", 
                color="quality",
                title=f"Rozkład {y_violin} w zależności od oceny"
            )
            st.plotly_chart(fig_violin, use_container_width=True)

        # METODA 3: Współrzędne Równoległe (Parallel Coordinates)
        with col_v2:
            st.subheader("3. Współrzędne Równoległe")
            st.caption("Śledź linię, aby zobaczyć profil chemiczny dobrych win (jasne kolory).")
            
            # Wybieramy tylko kilka kluczowych kolumn dla czytelności
            cols_parallel = ['alcohol', 'sulphates', 'volatile acidity', 'pH', 'quality']
            
            fig_par = px.parallel_coordinates(
                df_red[cols_parallel], 
                color="quality",
                labels={"alcohol": "Alkohol", "sulphates": "Siarczany", "quality": "Jakość"},
                color_continuous_scale=px.colors.diverging.Tealrose,
                color_continuous_midpoint=5.5,
                title="Profil chemiczny wina"
            )
            st.plotly_chart(fig_par, use_container_width=True)

    # --- ZAKŁADKA 2: MODELE ML (XGBOOST) ---
    with tab2:
        st.header("Trening i Predykcja")
        
        col_sett1, col_sett2 = st.columns([1, 3])
        
        with col_sett1:
            st.subheader("Ustawienia modelu")
            model_type = st.radio("Wybierz algorytm:", ["XGBoost (Gradient Boosting)", "Random Forest"])
            test_size = st.slider("Wielkość zbioru testowego (%)", 10, 40, 20) / 100.0
            
        with col_sett2:
            # Przygotowanie danych
            X = df_red.drop('quality', axis=1)
            y = df_red['quality']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Wybór modelu
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                # XGBoost
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
            
            # Trening
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metryki
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            st.write(f"### Wyniki dla modelu: {model_type}")
            m1, m2, m3 = st.columns(3)
            m1.metric("R2 Score", f"{r2:.3f}", help="Im bliżej 1.0, tym lepiej")
            m2.metric("MAE (Błąd średni)", f"{mae:.3f}", help="Średnia pomyłka w ocenie")
            m3.metric("RMSE", f"{rmse:.3f}", help="Pierwiastek błędu średniokwadratowego")
            
            # Feature Importance
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Cecha': X.columns, 'Waga': importances}).sort_values(by='Waga', ascending=True)
            
            fig_imp = px.bar(feat_df, x='Waga', y='Cecha', orientation='h', title="Wpływ cech na jakość wina")
            st.plotly_chart(fig_imp, use_container_width=True, height=300)

        st.divider()
        st.subheader("🧪 Symulator wina")
        
        # Inputy użytkownika
        c1, c2, c3, c4 = st.columns(4)
        val_alc = c1.slider("Alkohol (%)", 8.0, 15.0, 10.5)
        val_sul = c2.slider("Siarczany", 0.3, 2.0, 0.65)
        val_vol = c3.slider("Kwasowość lotna", 0.1, 1.6, 0.5)
        val_cit = c4.slider("Kwas cytrynowy", 0.0, 1.0, 0.25)
        
        # Uzupełnienie reszty średnimi
        input_data = pd.DataFrame([X.mean().values], columns=X.columns)
        input_data['alcohol'] = val_alc
        input_data['sulphates'] = val_sul
        input_data['volatile acidity'] = val_vol
        input_data['citric acid'] = val_cit
        
        if st.button("Oceń jakość (Użyj wytrenowanego modelu)"):
            prediction = model.predict(input_data)[0]
            st.success(f"Przewidywana jakość ({model_type}): **{prediction:.2f} / 10**")
            
            # Wizualizacja wyniku na "gauge chart" (licznik)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Jakość"},
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

    # --- ZAKŁADKA 3: SOMMELIER ---
    with tab3:
        st.header("Sommelier (Baza wiedzy)")
        search = st.text_input("Szukaj potrawy lub wina:")
        if search:
            res = df_pair[df_pair.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)]
            st.dataframe(res[['wine_type', 'food_item', 'pairing_quality', 'description']], hide_index=True)
        else:
            st.info("Wpisz frazę powyżej.")
            st.dataframe(df_pair.head(10))

else:
    st.error("Brak plików danych!")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        u1 = st.file_uploader("Wgraj winequality-red.csv", type='csv')
    with col_u2:
        u2 = st.file_uploader("Wgraj wine_food_pairings.csv", type='csv')
        
    if u1 and u2:
        df_red = pd.read_csv(u1)
        df_pair = pd.read_csv(u2)
        st.rerun()
