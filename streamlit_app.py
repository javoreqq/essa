import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Konfiguracja strony
st.set_page_config(page_title="Wino i Jedzenie - Analiza", layout="wide")

# Funkcja do ładowania danych
@st.cache_data
def load_data():
    # Próba wczytania automatycznego
    try:
        df_quality = pd.read_csv('winequality-red.csv')
        df_pairing = pd.read_csv('wine_food_pairings.csv')
        return df_quality, df_pairing
    except FileNotFoundError:
        # Jeśli plików nie ma, pozwól użytkownikowi je wgrać
        st.warning("⚠️ Nie znaleziono plików CSV w folderze aplikacji.")
        st.markdown("Proszę wgrać je ręcznie poniżej:")
        
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Wgraj winequality-red.csv", type='csv')
        with col2:
            file2 = st.file_uploader("Wgraj wine_food_pairings.csv", type='csv')
            
        if file1 and file2:
            df_quality = pd.read_csv(file1)
            df_pairing = pd.read_csv(file2)
            return df_quality, df_pairing
        else:
            return None, None

    # Tworzenie zakładek
    tab1, tab2 = st.tabs(["📊 Analiza Jakości (Chemia)", "🍽️ Wirtualny Sommelier (Parowanie)"])

    # --- ZAKŁADKA 1: ANALIZA JAKOŚCI ---
    with tab1:
        st.header("Analiza czynników wpływających na jakość wina")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Przegląd danych")
            st.write(f"Liczba próbek: {df_red.shape[0]}")
            if st.checkbox("Pokaż surowe dane"):
                st.dataframe(df_red.head(10))
            
            st.markdown("### Statystyki")
            st.write(df_red.describe())

        with col2:
            st.subheader("Korelacje")
            st.markdown("Sprawdź, które parametry chemiczne są ze sobą powiązane.")
            
            # Mapa ciepła korelacji
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_red.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr, linewidths=0.5)
            st.pyplot(fig_corr)

        st.divider()
        
        st.subheader("Wpływ parametrów na ocenę jakości (Quality)")
        st.markdown("Wybierz parametr, aby zobaczyć jak rozkłada się w zależności od oceny jakości wina (skala 3-8).")
        
        # Wybór kolumny do analizy (bez kolumny 'quality')
        features = [col for col in df_red.columns if col != 'quality']
        selected_feature = st.selectbox("Wybierz parametr chemiczny:", features, index=features.index('alcohol'))
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Boxplot
            fig_box, ax_box = plt.subplots()
            sns.boxplot(data=df_red, x='quality', y=selected_feature, palette='Reds', ax=ax_box)
            ax_box.set_title(f"Rozkład: {selected_feature} vs Quality")
            st.pyplot(fig_box)
            
        with col_chart2:
            # Wyjaśnienie dla użytkownika
            st.info(f"Wybrano: **{selected_feature}**")
            avg_val = df_red.groupby('quality')[selected_feature].mean()
            st.write("Średnia wartość parametru dla każdej oceny jakości:")
            st.dataframe(avg_val.to_frame(name=f"Średnia {selected_feature}").T)

    # --- ZAKŁADKA 2: SOMMELIER ---
    with tab2:
        st.header("Dobieranie wina do potraw (i odwrotnie)")
        
        mode = st.radio("Co chcesz zrobić?", ["Mam jedzenie, szukam wina", "Mam wino, szukam jedzenia"])
        
        if mode == "Mam jedzenie, szukam wina":
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                cuisines = sorted(df_pair['cuisine'].unique())
                selected_cuisine = st.selectbox("Wybierz kuchnię:", ["Wszystkie"] + cuisines)
            
            with col_filter2:
                # Filtrowanie listy potraw na podstawie kuchni
                if selected_cuisine != "Wszystkie":
                    available_foods = sorted(df_pair[df_pair['cuisine'] == selected_cuisine]['food_item'].unique())
                else:
                    available_foods = sorted(df_pair['food_item'].unique())
                
                selected_food = st.selectbox("Wybierz konkretne danie:", available_foods)
            
            # Wyszukiwanie
            if st.button("Szukaj wina"):
                results = df_pair[df_pair['food_item'] == selected_food].sort_values(by='pairing_quality', ascending=False)
                
                if not results.empty:
                    st.success(f"Znaleziono propozycje dla: **{selected_food}**")
                    
                    # Wyświetlanie wyników w ładniejszej formie
                    for _, row in results.iterrows():
                        with st.expander(f"{row['wine_type']} ({row['wine_category']}) - Ocena: {row['quality_label']}"):
                            st.write(f"**Ocena numeryczna:** {row['pairing_quality']}/5")
                            st.write(f"**Opis:** {row['description']}")
                            if row['pairing_quality'] >= 4:
                                st.markdown("✅ *Rekomendowane połączenie*")
                            elif row['pairing_quality'] <= 2:
                                st.markdown("⚠️ *Odradzane połączenie*")
                else:
                    st.warning("Brak danych dla wybranego dania.")

        elif mode == "Mam wino, szukam jedzenia":
            col_wine1, col_wine2 = st.columns(2)
            
            with col_wine1:
                categories = sorted(df_pair['wine_category'].unique())
                selected_category = st.selectbox("Kategoria wina:", ["Wszystkie"] + categories)
            
            with col_wine2:
                if selected_category != "Wszystkie":
                    wine_types = sorted(df_pair[df_pair['wine_category'] == selected_category]['wine_type'].unique())
                else:
                    wine_types = sorted(df_pair['wine_type'].unique())
                
                selected_wine = st.selectbox("Typ wina:", wine_types)
                
            if st.button("Szukaj potraw"):
                # Szukamy tylko dobrych połączeń (ocena >= 4)
                results = df_pair[
                    (df_pair['wine_type'] == selected_wine) & 
                    (df_pair['pairing_quality'] >= 4)
                ].sort_values(by='pairing_quality', ascending=False)
                
                if not results.empty:
                    st.success(f"Najlepsze potrawy do: **{selected_wine}**")
                    st.dataframe(results[['food_item', 'food_category', 'cuisine', 'pairing_quality', 'description']], hide_index=True)
                else:
                    st.warning("Nie znaleziono wybitnych połączeń ('Excellent'/'Good') w bazie dla tego wina. Spróbuj innego typu.")

    # Stopka
    st.sidebar.markdown("---")
    st.sidebar.info("Aplikacja stworzona na podstawie datasetów Wine Quality & Food Pairings.")
    
else:
    st.stop()
