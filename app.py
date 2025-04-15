import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Wczytanie modelu
def load_model(filename):
    with open(filename, "rb") as file:
        model_tuple = pickle.load(file)
    return model_tuple[1]

logistic_regression = load_model("Prediction/Logistic_Regression.pkl")

# Wczytanie pipeline'u transformacji
with open("Prediction/transformation_pipeline.pkl", "rb") as f:
    transformation_pipeline = pickle.load(f)
# Wczytanie explainera SHAP
with open("Prediction/shap_explainer.pkl", "rb") as f:
    loaded_explainer = pickle.load(f)
# =============================================================================
# WPROWADZANIE DANYCH PACJENTA (SIDEBAR)
# =============================================================================

def sidebar_inputs():
    with st.sidebar:
        st.info(
            "**Skorzystaj z formularza, aby wprowadzić dane i rozpocząć analizę**"
        )
    # Płeć

    sex = st.sidebar.selectbox("Wybierz płeć:", ["Mężczyzna", "Kobieta"])
    sex_val = "M" if sex == "Mężczyzna" else "F"

    # Wiek
    age = st.sidebar.slider(
        "Podaj wiek (lata):",
        min_value=28,
        max_value=77,
        value=40
    )


    chest_pain_type = st.sidebar.selectbox(
        "Wybierz rodzaj bólu w klatce piersiowej:",
        [
            "Typowa dławica piersiowa",
            "Atypowa dławica",
            "Ból nieanginowy",
            "Brak objawów"
        ]
    )

    # Spoczynkowe ciśnienie skurczowe krwi
    resting_bp = st.sidebar.slider(
        "Podaj spoczynkowe ciśnienie skurczowe krwi (mm Hg):",
        80,
        200,
        120
    )

    # Poziom cholesterolu
    cholesterol = st.sidebar.slider(
        "Podaj poziom cholesterolu całkowitego (mg/dl):",
        85,
        600,
        200
    )

    # Poziom cukru we krwi
    fasting_bs_option = st.sidebar.selectbox(
        "Określ, czy poziom cukru we krwi na czczo przekracza 120 mg/dl:",
        ["Nie (≤120)", "Tak (>120)"]
    )
    fasting_bs = 1 if "Tak" in fasting_bs_option else 0


    resting_ecg = st.sidebar.selectbox(
        "Wybierz wynik badania EKG (elektrokardiogramu) w spoczynku:",
        [
            "Prawidłowy zapis EKG",
            "Zmiany w odcinku ST–T",
            "Przerost lewej komory serca"
        ]
    )

    # Maksymalna częstość akcji serca
    max_hr = st.sidebar.slider(
        "Podaj maksymalną częstość akcji serca (uderzeń/min):",
        60,
        200,
        150
    )

    # Dławica wysiłkowa
    exercise_angina = st.sidebar.selectbox(
        "Określ, czy występuje ból w klatce piersiowej podczas wysiłku:",
        ["Nie", "Tak"]
    )


    # Oldpeak (obniżenie odcinka ST w zapisie EKG)
    oldpeak = st.sidebar.slider(
        "Podaj wartość oldpeak (obniżenie ST) w zapisie EKG (mV):",
        0.0,
        6.2,
        1.0,
        step=0.1
    )


    st_slope = st.sidebar.selectbox(
        "Określa, w jaki sposób kształtuje się nachylenie odcinka ST w EKG:",
        ["W górę", "Płaskie", "W dół"]
    )

    inputs = {
        "Sex": sex_val,
        "Age": age,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    return inputs
# =============================================================================
#  Funkcja tworząca DataFrame z wartościami wejściowymi (z sitebar input)
#  dodatkowo mapowanie wartości na oryginalne nazwy (po angielsku)
# =============================================================================
def create_input_dataframe(inputs: dict) -> pd.DataFrame:
    # Mapa dla płci (Sex):
    sex_str = inputs["Sex"]

    # Mapa dla ChestPainType:
    chest_pain_map = {
        "Typowa dławica piersiowa": "TA",
        "Atypowa dławica": "ATA",
        "Ból nieanginowy": "NAP",
        "Brak objawów": "ASY"
    }
    chest_pain_code = chest_pain_map[inputs["ChestPainType"]]

    # Mapa dla RestingECG:
    resting_ecg_map = {
        "Prawidłowy zapis EKG": "Normal",
        "Zmiany w odcinku ST–T": "ST",
        "Przerost lewej komory serca": "LVH"
    }
    resting_ecg_code = resting_ecg_map[inputs["RestingECG"]]

    # Mapa dla ExerciseAngina:
    exercise_angina_map = {
        "Nie": "N",
        "Tak": "Y"
    }
    exercise_angina_code = exercise_angina_map[inputs["ExerciseAngina"]]

    # Mapa dla ST_Slope:
    st_slope_map = {
        "W górę": "Up",
        "Płaskie": "Flat",
        "W dół": "Down"
    }
    st_slope_code = st_slope_map[inputs["ST_Slope"]]

    # Składamy wiersz do DataFrame
    row = {
        "Age": inputs["Age"], # liczba
        "Sex": sex_str,  # "M"/"F"
        "ChestPainType": chest_pain_code,  # "TA","ATA","NAP","ASY"
        "RestingBP": inputs["RestingBP"],  # liczba
        "Cholesterol": inputs["Cholesterol"],  # liczba
        "FastingBS": inputs["FastingBS"],  # 0 lub 1
        "RestingECG": resting_ecg_code,  # "Normal","ST","LVH"
        "MaxHR": inputs["MaxHR"],  # liczba
        "ExerciseAngina": exercise_angina_code,  # "N" lub "Y"
        "Oldpeak": inputs["Oldpeak"],  # liczba
        "ST_Slope": st_slope_code  # "Up","Flat","Down"
    }

    return pd.DataFrame([row])
# =============================================================================
#  ZAKŁADKA STRONY GŁÓWNEJ
# =============================================================================
def page_home(inputs):
    # Sekcja powitalna
    st.markdown("""
    <div class="hero-container">
      <div class="hero-text">
        <h1 class="hero-title">CardioPredict: Inteligentna Diagnostyka Chorób Serca</h1>
        <div class="hero-subtitle">
        <b>CardioPredict</b> to nowoczesne narzędzie wspierające diagnostykę i profilaktykę chorób serca, wykorzystujące sztuczną inteligencję.  
        Aplikacja analizuje kluczowe parametry zdrowotne pacjenta i przy użyciu zaawansowanych algorytmów uczenia maszynowego ocenia ryzyko wystąpienia chorób sercowo-naczyniowych.  
        Dzięki <b>intuicyjnemu interfejsowi, szczegółowym wizualizacjom oraz przejrzystej interpretacji wyników</b>, użytkownik może lepiej zrozumieć swój stan zdrowia i podjąć świadome decyzje dotyczące profilaktyki.
        </div>

   
   
   """, unsafe_allow_html=True)

    # Karty
    st.markdown("""
    <h3>🔍 Główne zakładki aplikacji</h3>

    <div class="cards-container">

    <div class="card">
      <h4>🩺 Predykcja</h4>
      <p>Na podstawie wprowadzonych parametrów zdrowotnych aplikacja dokonuje predykcji ryzyka choroby serca za pomocą modelu uczenia maszynowego. Dodatkowo wizualizuje wpływ poszczególnych cech na wynik za pomocą wykresu.</p>
    </div>

    <div class="card">
      <h4>📂 Import CSV</h4>
      <p>Możliwość przetwarzania wielu rekordów jednocześnie poprzez załadowanie pliku CSV. Aplikacja analizuje dane każdego pacjenta i zwraca plik z dodaną kolumną wskazującą przewidywane ryzyko choroby serca.</p>
    </div>

    <div class="card">
      <h4>📊 Wizualizacje</h4>
      <p>Interaktywna analiza danych pacjenta na tle populacji. Dostępne są wykresy radarowe, histogramy, wykresy pudełkowe, kołowe oraz skumulowane słupkowe, umożliwiające ocenę wartości pacjenta w porównaniu do trendów populacyjnych.</p>
    </div>

    <div class="card">
      <h4>📈 Analiza wyników</h4>
      <p>Porównanie wyników pacjenta z normami medycznymi oraz wskazanie wartości poza zakresem. Dodatkowo informacja o korelacji poszczególnych parametrów na wynik predykcji i interpretacja znaczenia wyników.</p>
    </div>

    <div class="card">
      <h4>🎯 Skuteczność predykcji</h4>
      <p>Ocena skuteczności modeli uczenia maszynowego na podstawie wskaźników dokładności, macierzy pomyłek i AUC-ROC. Prezentacja porównań modeli oraz analiza kluczowych cech wpływających na predykcję.</p>
    </div>

    <div class="card">
      <h4>📚 Dokumentacja projektu</h4>
      <p>Dokumentacja zawiera informacje o założeniach projektu, źródle danych oraz wiarygodności wykorzystanego zbioru z platformy Kaggle.</p>

      </div>


    </div>
    """, unsafe_allow_html=True)


    #st.write(inputs)
    #st.write(create_input_dataframe(inputs))
# =============================================================================
#  ZAKŁADKA STRONY PREDYKCJI
# =============================================================================
def page_prediction(inputs):
    col_1, col_2 = st.columns([3, 1])
    with col_1:
        st.title("🩺 Predykcja Choroby Serca")
        st.markdown("""
        ## 🔍 Jak działa predykcja?
        Model uczenia maszynowego analizuje wprowadzone przez Ciebie dane zdrowotne i na ich podstawie **oszacowuje ryzyko choroby serca**. Predykcja opiera się na wzorcach wykrytych w dużych zbiorach danych pacjentów z problemami kardiologicznymi.

        Po wprowadzeniu swoich parametrów zdrowotnych i wybraniu modelu otrzymasz:
        - **Wynik predykcji**: informacja, czy istnieje podwyższone ryzyko choroby serca.
        - **Prawdopodobieństwo**: określa stopień pewności modelu co do swojej prognozy.
        - **Wizualizacje** wykresy SHAP pokazujące wpływ poszczególnych parametrów na wynik predykcji.
        """)
    with col_2:
        st.image("assets/heart.jpg", width=250,
                 caption="Źródło: [Unsplash](https://unsplash.com/photos/orange-heart-decor-NIuGLCC7q54)")

    # Inicjujemy lokalne zmienne
    prediction = None
    prob = None
    csv_data = None
    shap_fig = None

    def get_gauge_chart(prob: float):
        """
        Zwraca wykres wskaźnikowy (Gauge Chart) z wartością procentową.
        Jeśli prob jest None lub poza zakresem [0,1], ustawia 0 jako wartość domyślną.
        """
        if prob is None or not isinstance(prob, (float, int)) or np.isnan(prob) or prob < 0 or prob > 1:
            st.warning("⚠️ Nieprawidłowa wartość prawdopodobieństwa. Ustawiono 0%.")
            prob = 0.0

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", "font": {"size": 70}},
            title={'text': "RYZYKO CHOROBY SERCA", "font": {"size": 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prob >= 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "orange"}
                ]
            }
        ))
        fig.update_layout(height=350)
        return fig

    def generate_csv(df_input, prediction, probability):
        """
        Tworzy plik CSV z informacją o predykcji i prawdopodobieństwie.
        """
        df_csv = df_input.copy()
        df_csv["Predykcja"] = "TAK" if prediction == 1 else "NIE"
        df_csv["Prawdopodobieństwo"] = f"{probability * 100:.2f}%" if probability is not None else "Brak danych"

        csv_modif = df_csv.to_csv(index=False, sep=",").encode("utf-8")

        return csv_modif

    def translate_features(X_transformed):
        """
        Tłumaczy nazwy kolumn na język polski, aby SHAP Waterfall wyświetlał przyjazne nazwy.
        """
        feature_translation = {
            "Age": "Wiek",
            "Sex_M": "Płeć",
            "ChestPainType_TA": "Ból w klatce: Typowa dławica",
            "ChestPainType_ATA": "Ból w klatce: Atypowa dławica",
            "ChestPainType_NAP": "Ból w klatce: Nieanginowy",
            "RestingBP": "Ciśnienie skurczowe",
            "Cholesterol": "Poziom cholesterolu",
            "FastingBS_1": "Cukier we krwi na czczo",
            "RestingECG_Normal": "EKG: Prawidłowy zapis",
            "RestingECG_ST": "EKG: Nieprawidłowości ST-T",
            "MaxHR": "Maksymalne tętno",
            "ExerciseAngina_Y": "Dławica wysiłkowa",
            "Oldpeak": "Depresja ST",
            "ST_Slope_Up": "Nachylenie ST: W górę",
            "ST_Slope_Flat": "Nachylenie ST: Płaskie"
        }
        new_columns = transformation_pipeline.named_steps['preprocessor'].get_feature_names_out()
        df = pd.DataFrame(X_transformed, columns=new_columns)
        df.columns = [feature_translation.get(col, col) for col in df.columns]
        return df

    # --- Logika po kliknięciu przycisku ---
    if st.button("🔄 Oblicz predykcję"):
        # Tworzymy DataFrame z danymi wejściowymi
        df_input = create_input_dataframe(inputs)

        # Transformacja danych
        X_transformed = transformation_pipeline.transform(df_input)

        # Predykcja (0 lub 1)
        prediction = logistic_regression.predict(X_transformed)[0]

        # Prawdopodobieństwo predykcji klasy 1
        prob = logistic_regression.predict_proba(X_transformed)[0][1]

        # Tworzenie CSV z danymi pacjenta
        csv_data = generate_csv(df_input, prediction, prob)

        # Obliczanie wartości SHAP
        try:
            new_patient_df = translate_features(X_transformed)
            shap_values_new_patient = loaded_explainer(new_patient_df)
            shap_values_new_patient_class_1 = shap_values_new_patient[..., 1]

            # Generowanie wykresu SHAP Waterfall
            fig_shap, ax = plt.subplots(figsize=(6, 3))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_new_patient_class_1[0],
                    base_values=shap_values_new_patient.base_values[0],
                    data=new_patient_df.iloc[0],
                    feature_names=new_patient_df.columns
                ),
                show=False
            )
            shap_fig = fig_shap
        except Exception as e:
            st.error(f"❌ Błąd podczas generowania wykresu SHAP: {e}")
            shap_fig = None

    # --- Wyświetlanie wyniku ---
    if prob is not None:
        col_chart, col_text = st.columns([1, 2])
        with col_chart:
            # Zawsze rysujemy gauge chart - nawet przy kolejnych re-runach
            gauge_fig = get_gauge_chart(prob)
            st.plotly_chart(gauge_fig)

            # Przyciski pobierania CSV
            if csv_data is not None:
                st.download_button(
                    label="📥 Pobierz wynik jako CSV",
                    data=csv_data,
                    file_name="prediction.csv",
                    mime="text/csv"
                )
        with col_text:
            if prediction == 1:
                st.markdown("""
                ## ⚠️ **Wynik: Podwyższone ryzyko choroby serca**

                Na podstawie Twoich danych model szacuje **istotne prawdopodobieństwo wystąpienia choroby sercowo-naczyniowej**.  
                Oznacza to, że Twój profil zdrowotny ma cechy charakterystyczne dla osób, u których zdiagnozowano chorobę serca.

                ### **Co to oznacza w praktyce?**
                - Model porównuje Twoje parametry z danymi wielu innych pacjentów i wskazuje, że istnieją podobieństwa do przypadków, gdzie potwierdzono chorobę serca.
                - Parametry mogą obejmować m.in. wiek, ciśnienie krwi, poziom cholesterolu czy wyniki EKG, a każdy z nich ma określony wpływ na końcową decyzję.

                ### **Dlaczego jest to istotne?**
                Wcześniejsze wykrycie zagrożenia umożliwia podjęcie kroków profilaktycznych:  
                - Zmianę stylu życia 
                - Dalszą diagnostykę.                  
                """)
            else:
                st.markdown("""
                ## ✅ **Wynik: Brak podwyższonego ryzyka choroby serca**

                Na podstawie wprowadzonych danych model nie wykrywa istotnych sygnałów mogących wskazywać na podwyższone ryzyko choroby sercowo-naczyniowej.  
                Oznacza to, że parametry Twojego profilu zdrowotnego przypominają dane osób, u których choroba serca nie wystąpiła.

                ### **Co to oznacza w praktyce?**
                - Model analizuje m.in. Twoje wyniki EKG, poziomy ciśnienia krwi, cholesterolu oraz inne cechy, zestawiając je z dużą bazą danych.
                - Uzyskany wynik sugeruje, że aktualnie nie ma wyraźnych przesłanek do uznania Twojego stanu za zagrożony.

                ### **Dlaczego jest to istotne?**
                Nawet jeśli aktualne wyniki wskazują na brak podwyższonego ryzyka:  
                - Warto dbać o profilaktykę, zdrową dietę i aktywność fizyczną.  
                - Zalecane są okresowe badania kontrolne, aby utrzymać dobry stan zdrowia i wcześnie wykrywać ewentualne zmiany.
                """)
            st.markdown(f"### **Szacowane prawdopodobieństwo choroby serca: {prob * 100:.1f}%**")
            st.markdown("""
                            Im wyższy procent, tym większe prawdopodobieństwo, że pacjent może mieć problemy sercowe.                  
                            """)
        st.subheader("Interpretacja wyniku: wpływ cech na predykcję (SHAP Waterfall)")
        with st.expander("ℹ️ Jak interpretować wykres SHAP Waterfall?", expanded=False):
            st.markdown("""
                 ### 🔍 **Co przedstawia wykres SHAP Waterfall?**
            Wykres SHAP Waterfall pokazuje, jak poszczególne cechy wpłynęły na końcowy wynik modelu.  
            Oś pozioma to wartość predykcji, a poszczególne paski reprezentują wpływ cech:

            - **Czerwone paski**🔴 oznaczają cechy, które zwiększyły prawdopodobieństwo choroby.  
            - **Niebieskie paski**🔵 oznaczają cechy, które je zmniejszyły.  
            - **Wartość bazowa** E[f(X)] to średnia predykcja modelu dla całej populacji. 
            - **f(x)** to wartość przewidywania modelu dla konkretnego przypadku, która w przypadku klasyfikacji jest **prawdopodobieństwem**.

            Wartość końcowa powstaje jako suma wartości SHAP i wartości bazowej.  

            ### ⚠ **Dlaczego niektóre cechy mogą nie być widoczne?**  
            Model wykorzystuje one-hot encoding z drop_first=True, co oznacza, że jedna kategoria w każdej grupie  
            jest pomijana i traktowana jako wartość domyślna. Jeśli wybrana wartość pacjenta była usuniętą kategorią,  
            nie pojawi się na wykresie, ale jest brana pod uwagę w wartości bazowej.  

            ### 📊 **Jak interpretować wykres?**  
            🔹 Im dłuższy pasek, tym większy wpływ cechy na predykcję.  
            🔹 Jeśli jakaś cecha nie pojawia się na wykresie, oznacza to, że jej wpływ był minimalny lub została zakodowana jako domyślna wartość.  
            🔹 Wynik modelu powstaje poprzez stopniowe dodawanie i odejmowanie wpływów cech do wartości bazowej.
            """)
        # Wyświetlenie wykresu SHAP (jeśli udało się go wygenerować)
        if shap_fig is not None:
            emp1, shap_waterfall, emp2 = st.columns([1, 20, 1])
            with shap_waterfall:
                st.pyplot(shap_fig)
        else:
            st.warning("⚠️ Wykres SHAP Waterfall jest niedostępny.")
    else:
        st.info("ℹ️ Kliknij **Oblicz predykcję**, aby zobaczyć wynik i wykresy.")
# =============================================================================
#  ZAKŁADKA STRONY PREDYKCJI MASOWEJ
# =============================================================================
def page_mass_prediction():
    st.title("Analiza danych pacjentów - Predykcja ryzyka chorób serca")

    st.info("Możesz tu przeprowadzić przewidywanie ryzyka choroby serca **dla wielu osób jednocześnie**, wysyłając plik CSV z danymi pacjentów.\n\n"
            "Następnie **po przesłaniu otrzymasz wyniki** do pobrania w formie pilku CSV z dodaną kolumną.\n\n")


    with st.expander("ℹ️ Pokaż instrukcje dotyczące pliku CSV"): # expanded=True
        st.markdown("""
        
        ### **Instrukcja dla użytkownika**

        Aby skorzystać z aplikacji, wgraj plik **CSV** zawierający dane pacjentów zgodnie z poniższą specyfikacją:

        ---

        ### **Wymagane kolumny w pliku CSV:**

        | **Atrybut (ENG / PL)**      | **Opis (znaczenie + jednostka)**                     |
        |-----------------------------|-----------------------------------------------------|
        | **Age / Wiek**              | Wiek pacjenta w latach **[lata]**                   |
        | **Sex / Płeć**              | Płeć pacjenta: `M` – mężczyzna, `F` – kobieta **[kategoria]** |
        | **ChestPainType / Rodzaj bólu w klatce piersiowej** | Typ bólu w klatce piersiowej: `TA` – typowa dławica, `ATA` – atypowa dławica, `NAP` – ból nieanginowy, `ASY` – brak objawów **[kategoria]** |
        | **RestingBP / Spoczynkowe ciśnienie krwi** | Skurczowe ciśnienie krwi zmierzone w spoczynku **[mm Hg]** |
        | **Cholesterol / Cholesterol całkowity** | Poziom cholesterolu całkowitego we krwi **[mg/dl]** |
        | **FastingBS / Cukier we krwi na czczo** | Czy poziom glukozy na czczo przekracza 120 mg/dl: `1` – tak, `0` – nie **[kategoria]** |
        | **RestingECG / Elektrokardiogram spoczynkowy** | Wynik badania EKG w spoczynku: `Normal` – prawidłowy, `ST` – nieprawidłowości ST-T, `LVH` – przerost lewej komory **[kategoria]** |
        | **MaxHR / Maksymalne tętno** | Najwyższa wartość tętna pacjenta osiągnięta podczas testu wysiłkowego **[bpm]** |
        | **ExerciseAngina / Dławica wysiłkowa** | Czy pacjent odczuwa ból w klatce piersiowej podczas wysiłku: `Y` – tak, `N` – nie **[kategoria]** |
        | **Oldpeak / Obniżenie odcinka ST** | Stopień obniżenia odcinka ST w EKG (depresja ST) **[mV]** |
        | **ST_Slope / Nachylenie odcinka ST** | Charakterystyka nachylenia odcinka ST: `Up` – nachylenie w górę, `Flat` – płaski, `Down` – nachylenie w dół **[kategoria]** |

        ---

        **Przykład poprawnego pliku CSV:**

        ```
        Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
        45,M,TA,120,200,0,Normal,150,N,1.0,Up
        60,F,ATA,130,250,1,ST,140,Y,2.3,Flat
        ```

        **Uwaga:** Upewnij się, że:
        - Wszystkie wymagane kolumny są obecne.
        - Nie występują brakujące wartości.
        - Dane są zgodne z powyższym formatem.
        """, unsafe_allow_html=True)

    # Funkcja walidująca plik CSV
    def validate_csv(df):
        # Oczekiwana kolejność kolumn (nie sprawdzam nazw kolumn)
        expected_columns_count = 11  # Zakładamy, że plik ma 11 kolumn
        if df.shape[1] != expected_columns_count:
            return f"Błąd: Oczekiwano {expected_columns_count} kolumn, ale znaleziono {df.shape[1]}."

        # Sprawdzanie brakujących wartości
        if df.isnull().any().any():
            missing_values = df.isnull().sum()
            missing_values = {col: int(count) for col, count in missing_values.items() if count > 0}
            return f"Wykryto brakujące wartości w kolumnach: {missing_values}"

        # Sprawdzam dozwolone wartości w kolumnach kategorycznych
        allowed_sex = ['M', 'F']
        allowed_chest_pain_type = ['TA', 'ATA', 'NAP', 'ASY']
        allowed_resting_ecg = ['Normal', 'ST', 'LVH']
        allowed_exercise_angina = ['Y', 'N']
        allowed_st_slope = ['Up', 'Flat', 'Down']

        # Sprawdzanie kategorycznych kolumn
        if not df.iloc[:, 1].isin(allowed_sex).all():  # Sex - kolumna 1 (2. kolumna)
            return "Błąd: Nieprawidłowe wartości w kolumnie 'Sex'!"

        if not df.iloc[:, 2].isin(allowed_chest_pain_type).all():  # ChestPainType - kolumna 2 (3. kolumna)
            return "Błąd: Nieprawidłowe wartości w kolumnie 'ChestPainType'!"

        if not df.iloc[:, 6].isin(allowed_resting_ecg).all():  # RestingECG - kolumna 6 (7. kolumna)
            return "Błąd: Nieprawidłowe wartości w kolumnie 'RestingECG'!"

        if not df.iloc[:, 8].isin(allowed_exercise_angina).all():  # ExerciseAngina - kolumna 8 (9. kolumna)
            return "Błąd: Nieprawidłowe wartości w kolumnie 'ExerciseAngina'!"

        if not df.iloc[:, 10].isin(allowed_st_slope).all():  # ST_Slope - kolumna 10 (11. kolumna)
            return "Błąd: Nieprawidłowe wartości w kolumnie 'ST_Slope'!"

        # Sprawdzanie minimalnych wartości dla kolumn liczbowych
        # Age (0), RestingBP (3), Cholesterol (4), MaxHR (7), Oldpeak (9)

        if df.iloc[:, 0].min() <= 0:  # Age - kolumna 0 (1. kolumna)
            return "Błąd: Wartość 'Age' musi być większa od 0!"

        if df.iloc[:, 3].min() < 0:  # RestingBP - kolumna 3 (4. kolumna)
            return "Błąd: Wartość 'RestingBP' musi być większa od 0!"

        if df.iloc[:, 4].min() < 0:  # Cholesterol - kolumna 4 (5. kolumna)
            return "Błąd: Wartość 'Cholesterol' musi być większa od 0!"

        if df.iloc[:, 7].min() < 0:  # MaxHR - kolumna 7 (8. kolumna)
            return "Błąd: Wartość 'MaxHR' musi być większa od 0!"

        # if df.iloc[:, 9].min() < 0:  # Oldpeak - kolumna 9 (10. kolumna)
        #     return "Błąd: Wartość 'Oldpeak' nie może być mniejsza niż 0!"

        # Wszystkie testy przeszły pomyślnie
        return None

    uploaded_file = st.file_uploader("Załaduj plik CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Wczytanie danych z pliku
            df_csv = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Nie udało się wczytać pliku: {e}")
            return

        # Walidacja danych
        validation_error = validate_csv(df_csv)

        if validation_error:
            st.error(validation_error)
        else:
            st.success("Plik CSV został poprawnie wczytany i zwalidowany!")
            st.dataframe(df_csv.head(10))

            try:
                X_transformed = transformation_pipeline.transform(df_csv)
            except Exception as e:
                st.error(f"Problem z transformacją danych: {e}")
                return

            # Wykonanie predykcji
            preds = logistic_regression.predict(X_transformed)
            probs = logistic_regression.predict_proba(X_transformed)[:, 1]  # Prawdopodobieństwo klasy 1

            # Dodanie kolumny z wynikami predykcji i prawdopodobieństwem
            df_result = df_csv.copy()
            df_result["HeartDisease"] = preds
            df_result["probability"] = (probs * 100).round(2).astype(str) + "%"  # w formacie %

            st.success("Podgląd wyników:")
            st.dataframe(df_result.head(10))

            # Przygotowanie danych do pobrania
            csv_data = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Pobierz wynik jako CSV",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv"
            )
# =============================================================================
#  ZAKŁADKA STRONY DO WIZUALIZACJI
# =============================================================================
def page_visualizations(inputs):

    # -------------------------------
    # 1) Tytuł i krótki opis zakładki
    # -------------------------------
    st.title("Personalizowana analiza porównawcza pacjenta względem populacji")
    st.info(
        "Możesz tu **przeanalizować swoje wyniki w odniesieniu do populacji**, sprawdzając, jak Twoje parametry zdrowotne wypadają na tle ogólnych trendów.\n\n"
        "Dzięki interaktywnym wykresom **zyskasz lepsze zrozumienie** wpływu poszczególnych cech na wynik predykcji oraz dowiesz się, które wartości odbiegają od normy.\n"
    )

    # --------------------------------------------------------------
    # 2) Definiujemy listy cech ciągłych i kategorycznych oraz mapy
    # --------------------------------------------------------------
    cont_vars = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    cat_vars = ["Sex","FastingBS", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    cont_mappings = {
        "Age": "Wiek",
        "RestingBP": "Ciśnienie spoczynkowe",
        "Cholesterol": "Cholesterol",
        "MaxHR": "Maksymalny HR",
        "Oldpeak": "Obniżenie ST"
    }
    cat_mappings = {
        "FastingBS": {0: "poniżej 120", 1: "powyżej 120"},
        "Sex": {"M": "Mężczyzna", "F": "Kobieta"},
        "ChestPainType": {
            "TA": "Typowa dławica piersiowa",
            "ATA": "Atypowa dławica",
            "NAP": "Ból nieanginowy",
            "ASY": "Brak objawów"
        },
        "RestingECG": {
            "Normal": "Prawidłowy zapis EKG",
            "ST": "Zmiany w odcinku ST–T",
            "LVH": "Przerost lewej komory serca"
        },
        "ExerciseAngina": {"N": "Nie", "Y": "Tak"},
        "ST_Slope": {"Up": "W górę", "Flat": "Płaskie", "Down": "W dół"}
    }
    cat_label_mapping = {
        "Sex": "Płeć",
        "ChestPainType": "Ból w klatce piersiowej",
        "RestingECG": "Wynik EKG",
        "ExerciseAngina": "Dławica wysiłkowa",
        "ST_Slope": "Nachylenie ST",
        "FastingBS" : "Cukier we krwi czczo"
    }
    def get_radar_chart(data):
        """
        Funkcja tworząca wykres radarowy (Plotly) porównujący wartości pacjenta
        ze średnimi wartościami w populacji.
        """
        ranges = {
            "Age": (28, 77),
            "RestingBP": (80, 200),
            "Cholesterol": (85, 600),
            "MaxHR": (60, 220),
            "Oldpeak": (0, 6.2)
        }
        means = {
            "Age": 53,
            "RestingBP": 132,
            "Cholesterol": 243,
            "MaxHR": 136,
            "Oldpeak": 0.9
        }

        features = list(ranges.keys())
        theta_labels = [cont_mappings[feat] for feat in features]

        patient_vals = []
        mean_vals = []

        for feat in features:
            min_val, max_val = ranges[feat]
            # Normalizacja pacjenta
            patient_norm = (data[feat] - min_val) / (max_val - min_val)
            # Normalizacja średniej
            mean_norm = (means[feat] - min_val) / (max_val - min_val)
            patient_vals.append(patient_norm)
            mean_vals.append(mean_norm)

        fig = go.Figure()
        # Pacjent
        fig.add_trace(go.Scatterpolar(
            r=patient_vals,
            theta=theta_labels,
            fill='toself',
            name='Pacjent',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        # Średnia populacyjna
        fig.add_trace(go.Scatterpolar(
            r=mean_vals,
            theta=theta_labels,
            fill='toself',
            name='Średnia w populacji',
            line=dict(color='blue'),
            fillcolor='rgba(0,0,255,0.3)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            width=600,
            height=600,
            showlegend=True
        )
        return fig

    # -------------------------------------------------------
    # 4) Wyświetlenie wykresu radarowego (stała sekcja)
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    ### **Wykres radarowy** 

    Wykres radarowy pozwala na jednoczesne porównanie wielu parametrów zdrowotnych pacjenta 
    z wartościami średnimi dla całej populacji. Na wykresie wyróżniono wartości badanego pacjenta, 
    co ułatwia ocenę jego wyników w porównaniu do standardowych wartości w populacji.
    """)
    radar_fig = get_radar_chart(inputs)
    st.plotly_chart(radar_fig, use_container_width=True)
    # Tworzymy listę wszystkich zmiennych
    all_vars = cont_vars + cat_vars


    # -----------------------------------------------------------------------------------------
    # 5) Selectbox dla wszystkich zmiennych - NAD wykresem radarowym
    # -----------------------------------------------------------------------------------------

    selected_var = st.selectbox(
        "Wybierz parametr, który chcesz szczegółowo przeanalizować:",
        all_vars,
        format_func=lambda x: {
            "Age": "Wiek",
            "RestingBP": "Ciśnienie spoczynkowe",
            "Cholesterol": "Cholesterol",
            "MaxHR": "Maksymalny HR",
            "Oldpeak": "Obniżenie ST",
            "Sex": "Płeć",
            "ChestPainType": "Ból w klatce piersiowej",
            "RestingECG": "Wynik EKG",
            "FastingBS": "Cukier we krwi czczo",
            "ExerciseAngina": "Dławica wysiłkowa",
            "ST_Slope": "Nachylenie ST"

        }[x]
    )



    # -------------------------------------------------------
    # 6) Wczytanie danych z pliku "heart.csv"
    # -------------------------------------------------------
    try:
        df_heart = pd.read_csv("heart.csv")
    except Exception as e:
        st.error("Nie udało się wczytać danych z pliku heart.csv.")
        return

    # -------------------------------------------------------
    # 7) Wyświetlanie wykresów w zależności od typu zmiennej
    # -------------------------------------------------------
    if selected_var in cont_vars:
        # -----------------------
        # A) Zmienna ciągła
        # -----------------------
        st.markdown("---")
        st.markdown(f"## Szczegółowa analiza zmiennej: {cont_mappings[selected_var]}")
        st.markdown("""
        Pokazujemy tutaj histogram, wykres pudełkowy oraz rozkład zmiennej względem choroby serca 
        (wartość pacjenta zaznaczona czerwoną linią).
        """)

        patient_value = inputs[selected_var]

        # --- Histogram i boxplot ---
        fig_cont, axes_cont = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        # Lewy wykres: histogram + linia wartości pacjenta
        ax_hist = axes_cont[0]
        sns.histplot(
            data=df_heart,
            x=selected_var,
            kde=True,
            ax=ax_hist,
            color="#007EA7"
        )
        ax_hist.axvline(patient_value, color="red", linestyle="--", linewidth=2, label="Twoja wartość")
        ax_hist.set_title(f"Rozkład cechy: {cont_mappings[selected_var]}", fontsize=14)
        ax_hist.legend()

        # Prawy wykres: boxplot + linia wartości pacjenta
        ax_box = axes_cont[1]
        sns.boxplot(
            x=df_heart[selected_var],
            ax=ax_box,
            color="#007EA7"
        )
        ax_box.axvline(patient_value, color="red", linestyle="--", linewidth=2, label="Twoja wartość")
        ax_box.set_title(f"Wykres pudełkowy cechy: {cont_mappings[selected_var]}", fontsize=14)
        ax_box.legend()

        plt.tight_layout()
        st.pyplot(fig_cont)
#percentyl populacji
        percentyl = stats.percentileofscore(df_heart[selected_var], patient_value)
        st.markdown(f"## Twoja wartość znajduje się w *{percentyl:.0f}. percentylu* tej cechy w populacji.")

        # --- Dodatkowa analiza (histogram z hue="HeartDisease") ---
        st.markdown("---")
        st.markdown("""
        ### Rozkład wartości względem występowania choroby serca
        Poniżej przedstawiono wykres pokazujący rozkład wybranej cechy w zależności od obecności choroby serca w populacji, 
        wraz z zaznaczeniem Twojej wartości (czerwona linia).
        """)
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=df_heart,
            x=selected_var,
            hue="HeartDisease",
            multiple="stack",
            palette=["#69b3a2", "#d95f02"],
            edgecolor="black",
            kde=False
        )
        plt.axvline(patient_value, color="red", linestyle="--", linewidth=2, label="Twoja wartość")
        plt.title(f'Rozkład {cont_mappings[selected_var]} względem występowania choroby serca', fontsize=14)
        plt.xlabel(cont_mappings[selected_var])
        plt.ylabel("Liczba pacjentów")
        plt.legend(["Twoja wartość", "Brak choroby serca", "Choroba serca"])
        st.pyplot(plt.gcf())

    elif selected_var in cat_vars:
        # -----------------------
        # B) Zmienna kategoryczna
        # -----------------------
        st.markdown("---")
        st.markdown(f"## Szczegółowa analiza zmiennej: {cat_label_mapping[selected_var]}")
        st.markdown("""
        Pokazujemy tutaj **wykres kołowy** i **skumulowany wykres słupkowy** (z rozbiciem na chorobę serca),
        a także wyróżniamy wycinek lub słupek odpowiadający wartości pacjenta.
        """)

        # Wartość pacjenta
        raw_value = inputs[selected_var]
        translated_value = cat_mappings[selected_var].get(raw_value, raw_value)
        #.get(key, default) oznacza: jeśli klucz raw_value istnieje w mapowaniu, to zwróć jego wartość;
        # jeśli nie – zwróć oryginalną wartość (raw_value).

        # Przygotowanie do wykresu kołowego
        value_counts = df_heart[selected_var].value_counts()
        labels = [cat_mappings[selected_var].get(x, x) for x in value_counts.index]
        explode = [0.1 if lbl == translated_value else 0 for lbl in labels]

        # Przygotowanie danych do wykresu słupkowego (stacked)
        temp = (
            df_heart[[selected_var, 'HeartDisease']]
            .groupby([selected_var, 'HeartDisease'])
            .size()
            .unstack('HeartDisease', fill_value=0)
        )
        temp.rename(columns={0: 'Brak choroby', 1: 'Choroba'}, inplace=True)
        # Tłumaczymy indeks
        temp.index = [cat_mappings[selected_var].get(x, x) for x in temp.index]

        def stacked_barchart(data, ax, title='', ylabel=''):
            """Rysuje skumulowany wykres słupkowy."""
            data.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'],
                      alpha=0.85, edgecolor='black', ax=ax)

            ax.set_title(title, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel('Kategoria', fontsize=12)
            ax.tick_params(axis='x', labelrotation=0, labelsize=9)
            ax.tick_params(axis='y', labelsize=10)
            ax.legend(title="Choroba serca", fontsize=10)

            # Dodajemy adnotacje (procenty) na słupkach
            for i, idx in enumerate(data.index):
                total = data.loc[idx].sum()
                cum = 0
                for col in data.columns:
                    val = data.loc[idx, col]
                    pct = (val / total * 100) if total > 0 else 0
                    if val > 0:
                        ax.text(
                            i, cum + val / 2,
                            f'{pct:.1f}%',
                            ha='center', va='center', fontsize=10
                        )
                    cum += val

        # -----------------------------
        # Rysujemy wykresy obok siebie
        # -----------------------------
        fig_cat, (ax_pie, ax_bar) = plt.subplots(ncols=2, figsize=(14, 6))

        # --- WYKRES KOŁOWY ---
        ax_pie.pie(
            value_counts.values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            explode=explode
        )
        ax_pie.set_title(f'Wykres kołowy: {cat_label_mapping[selected_var]}', fontsize=14)

        # --- SKUMULOWANY WYKRES SŁUPKOWY ---
        stacked_barchart(
            temp,
            ax=ax_bar,
            title=f'{cat_label_mapping[selected_var]} vs. Choroba serca',
            ylabel='Liczebność'
        )

        # Dodajemy adnotację "Twoja wartość" nad odpowiednim słupkiem (jeśli istnieje w indeksie)
        if translated_value in temp.index:
            pos = list(temp.index).index(translated_value)
            total_val = temp.loc[translated_value].sum()
            max_val = temp.sum(axis=1).max()
            offset = max_val * 0.01
            annot_y = total_val + offset
            current_ylim = ax_bar.get_ylim()[1]
            if annot_y > current_ylim:
                annot_y = current_ylim * 0.99
            ax_bar.text(
                pos, annot_y, "Twoja wartość",
                ha='center', va='bottom', color="red",
                fontsize=10, fontweight="bold"
            )

        # Wyświetlamy połączoną figurę (pie + bar)
        plt.tight_layout()
        st.pyplot(fig_cat)
        plt.clf()

    else:
        st.warning("Wybrano nieznany typ zmiennej.")


# =============================================================================
#  ZAKŁADKA STRONY ANALIZY WYNIKÓW
# =============================================================================
def page_analysis(inputs):
    st.title("Podsumowanie profilu zdrowotnego")
    st.info(
        "Przegląd najważniejszych **parametrów** zdrowotnych wprowadzonych przez **użytkownika**.\n\n"
        "**Każda wartość została oceniona na podstawie przyjętych zakresów** i oznaczona jako prawidłowa, przeciętna lub odbiegająca od normy.\n"
    )


    analysis = {}

    # Płeć (1 = mężczyzna, 0 = kobieta)
    sex = inputs["Sex"]
    if sex == "M":
        analysis["Płeć"] = ("Mężczyzna – wyższe ryzyko chorób serca.", "warning")
    else:
        analysis["Płeć"] = ("Kobieta – niższe ryzyko chorób serca.", "info")

    # Wiek
    age = inputs["Age"]
    if age < 45:
        analysis["Wiek"] = (f"{age} lat – młody wiek, niższe ryzyko chorób serca.", "success")
    elif age < 60:
        analysis["Wiek"] = (f"{age} lat – umiarkowane ryzyko chorób serca.", "info")
    else:
        analysis["Wiek"] = (f"{age} lat – wyższe ryzyko chorób serca.", "warning")



    # Rodzaj bólu w klatce piersiowej
    chest = inputs["ChestPainType"]

    if chest == "Typowa dławica piersiowa":
        analysis["Rodzaj bólu w klatce piersiowej"] = (
            "Typowa dławica piersiowa - wysokie ryzyko!", "warning")  # Lub "danger"
    elif chest == "Atypowa dławica":
        analysis["Rodzaj bólu w klatce piersiowej"] = (
            "Atypowa dławica piersiowa - Umiarkowane ryzyko.", "info")
    elif chest == "Ból nieanginowy":
        analysis["Rodzaj bólu w klatce piersiowej"] = (
            "Ból nieanginowy - Mało prawdopodobne ryzyko", "info")
    else:
        analysis["Rodzaj bólu w klatce piersiowej"] = (
            "Brak objawów bólowych - Nie wyklucza choroby serca.", "info")

    # Ciśnienie spoczynkowe
    bp = inputs["RestingBP"]
    if bp < 120:
        analysis["Ciśnienie spoczynkowe"] = (f"{bp} mmHg – w normie.", "success")
    elif bp < 130:
        analysis["Ciśnienie spoczynkowe"] = (f"{bp} mmHg – lekko podwyższone.", "info")
    else:
        analysis["Ciśnienie spoczynkowe"] = (f"{bp} mmHg – podwyższone, zwiększa ryzyko.", "warning")

    # Cholesterol
    chol = inputs["Cholesterol"]
    if chol < 200:
        analysis["Cholesterol"] = (f"{chol} mg/dL – w normie.", "success")
    elif chol < 240:
        analysis["Cholesterol"] = (f"{chol} mg/dL – wynik graniczny.", "info")
    else:
        analysis["Cholesterol"] = (f"{chol} mg/dL – wysoki, zwiększa ryzyko chorób serca.", "warning")

    # Cukier we krwi na czczo
    fasting = inputs["FastingBS"]
    if fasting == 0:
        analysis["Cukier we krwi na czczo"] = ("W normie.", "success")
    else:
        analysis["Cukier we krwi na czczo"] = ("Podwyższony.", "warning")

    # EKG w spoczynku
    ecg = inputs["RestingECG"]
    if ecg == "Prawidłowy zapis EKG":
        analysis["EKG w spoczynku"] = ("Prawidłowy zapis EKG.", "success")
    elif ecg == "Zmiany w odcinku ST–T":
        analysis["EKG w spoczynku"] = ("Zmiany w odcinku ST–T – zwiększa ryzyko.", "warning")
    else:
        analysis["EKG w spoczynku"] = ("Przerost lewej komory – zwiększa ryzyko.", "warning")

    # Maksymalne tętno
    max_hr = inputs["MaxHR"]

    if max_hr >= 150:
        analysis["Maksymalny HR"] = (
            f"{max_hr} uderzeń/min – bardzo dobre tętno maksymalne.",
            "success"
        )
    elif 130 <= max_hr < 150:
        analysis["Maksymalny HR"] = (
            f"{max_hr} uderzeń/min – umiarkowany wynik.",
            "info"
        )
    else:
        analysis["Maksymalny HR"] = (
            f"{max_hr} uderzeń/min – niska wartość",
            "warning"
        )

    # Ból przy wysiłku
    ex_angina = inputs["ExerciseAngina"]
    if ex_angina == "Tak":
        analysis["Ból przy wysiłku"] = ("Występuje – zwiększa ryzyko chorób serca.", "warning")
    else:
        analysis["Ból przy wysiłku"] = ("Nie występuje – korzystny wynik.", "success")

    # Obniżenie ST
    oldpeak = inputs["Oldpeak"]
    if oldpeak < 1:
        analysis["Obniżenie ST"] = (f"{oldpeak} mV – w normie.", "success")
    elif oldpeak < 2:
        analysis["Obniżenie ST"] = (f"{oldpeak} mV – lekko podwyższone.", "info")
    else:
        analysis["Obniżenie ST"] = (f"{oldpeak} mV – wysoki, zwiększa ryzyko niedokrwienia.", "warning")

    # Nachylenie ST
    st_slope = inputs["ST_Slope"]
    if st_slope == "W górę":
        analysis["Nachylenie ST"] = ("W górę – typowy, korzystny wynik.", "success")
    elif st_slope == "Płaskie":
        analysis["Nachylenie ST"] = ("Płaskie – może wskazywać na pewne nieprawidłowości.", "info")
    else:
        analysis["Nachylenie ST"] = ("W dół – niepokojące, zwiększa ryzyko.", "warning")

    # Tworzymy zmienną html_table, w której zapisujemy kod HTML otwierający znacznik <table> oraz pierwszy wiersz (<tr>).
    # dajemy klasę analysis-table można ostylować w pliku css
    html_table = """
    <table class="analysis-table"> 
      <tr>
        <th>Parametr</th>
        <th>Ocena</th>
      </tr>
    """
    #Iterujemy przez słownik analysis, który zawiera klucz param (nazwę parametru) oraz krotkę (desc, level).
    #desc to tekst opisu lub oceny, a level to nazwa klasy (np. "success", "info", "warning"), która decyduje o kolorze wiersza.
    for param, (desc, level) in analysis.items():
        html_table += f"<tr class='{level}'><td><strong>{param}</strong></td><td><strong>{desc}</strong></td></tr>"
    html_table += "</table>"
    #wyświetlenie tabeli
    st.markdown(html_table, unsafe_allow_html=True)


# =============================================================================
#  ZAKŁADKA STRONY OCENA MODELI
# =============================================================================
def page_model_evaluation():
    st.title("📊 Skuteczność Predykcji")
    st.markdown("""
    W tej sekcji możesz przeanalizować skuteczność i charakterystykę używanych algorytmów uczenia maszynowego. 
    W projekcie wykorzystywany został model regresji logistycznej do predykcji, co zostało dodatkowo wsparte analizą korelacji oraz interpretacją decyzji modelu.
    """)

    # --- Sekcja 1: Analiza korelacji ---
    st.markdown("---")
    st.markdown("### Analiza korelacji")
    col_corr1, col_corr2 = st.columns([2, 1])
    with col_corr1:
        st.image("assets/corr.png", use_container_width=True)
    with col_corr2:
        st.markdown("""
**Wnioski z korelacji:**  
- **Dodatnie korelacje:** cechy takie jak płaskie nachylenie ST, dławica wysiłkowa i obniżenie ST silnie korelują z ryzykiem choroby serca.  
- **Ujemne korelacje:** większe wartości maksymalnego tętna oraz nachylenie ST w górę wskazują na niższe ryzyko.  

Korelacja nie oznacza przyczynowości, ale pomaga zidentyfikować kluczowe czynniki wpływające na ryzyko.
        """)

    # --- Sekcja 2: Interpretacja modelu regresji logistycznej ---
    st.markdown("---")
    st.markdown("### Interpretacja modelu regresji logistycznej")
    st.markdown("""
    Poniższy wykres SHAP przedstawia wpływ poszczególnych cech na wynik modelu regresji logistycznej.  
    Dzięki tej interpretacji możliwe jest zrozumienie, które cechy najbardziej przyczyniają się do przewidywania ryzyka.  
            """)
    col_inter1, col_inter2, = st.columns([1, 1])
    with col_inter1:
        # Placeholder – dostosuj ścieżkę według potrzeb
        st.image("assets/regression_features_importance.png", use_container_width=True)
    with col_inter2:
        st.image("assets/regression_features_importance_.png", use_container_width=True)
    with st.expander("Wyświetl wykres zależności dla poszczególnych cech"):
        st.image("assets/download.png", use_container_width=True)



    # Expander: Porównanie wykresów metryk modeli
    st.markdown("---")
    with st.expander("Porównanie metryk modeli"):
        st.markdown("#### Porównanie dokładności modeli")
        st.image("assets/acc.png", use_container_width=True,
                 caption="Wykres słupkowy – porównanie dokładności modeli ML.")
        st.markdown("#### Porównanie AUC-ROC")
        st.image("assets/auc_roc.png", use_container_width=True,
                 caption="Wykres słupkowy – porównanie AUC-ROC dla poszczególnych modeli.")
        st.markdown("#### Porównanie Precision, Recall i F1-score")
        st.image("assets/Prec_rec_f1.png", use_container_width=True,
                 caption="Zestawienie metryk precyzji, czułości i F1-score.")

    # Expander: Szczegółowa analiza poszczególnych modeli
    st.markdown("---")
    with st.expander("Szczegółowa analiza poszczególnych modeli"):
        st.markdown("Wybierz model, aby zobaczyć szczegółowe wyniki:")
        model_options = [
            "Logistic Regression",
            "Stacking Classifier",
            "Voting Classifier Soft",
            "Voting Classifier Hard",
            "SVM",
            "Random Forest",
            "KNN",
            "Decision Tree"
        ]
        selected_model = st.selectbox("Wybierz model:", model_options, index=0)
        if selected_model == "Logistic Regression":
            st.image("assets/Logistic_regresion_evaluation.png", use_container_width=True)
            st.markdown("""
**Regresja logistyczna** zapewnia stabilne wyniki oraz wysoką interpretowalność dzięki współczynnikom regresji.
            """)
        elif selected_model == "Stacking Classifier":
            st.image("assets/SC_evaluation.png", use_container_width=True)
            st.markdown("""
**Stacking Classifier** łączy wyniki wielu modeli bazowych, co przekłada się na wyższą generalizację.
            """)
        elif selected_model == "Voting Classifier Soft":
            st.image("assets/voting_cassifier_evaluation.png", use_container_width=True)
            st.markdown("""
**Voting Classifier Soft** oblicza średnie prawdopodobieństwa, osiągając wysoką precyzję.
            """)
        elif selected_model == "Voting Classifier Hard":
            st.image("assets/voting_classifier_hard_evaluation.png", use_container_width=True)
            st.markdown("""
**Voting Classifier Hard** stosuje zasadę większości głosów, choć nie obsługuje prognozowania prawdopodobieństw.
            """)
        elif selected_model == "SVM":
            st.image("assets/SVM_evaluation.png", use_container_width=True)
            st.markdown("""
**Support Vector Machine (SVM)** osiąga wysokie wyniki, choć wymaga precyzyjnego strojenia parametrów.
            """)
        elif selected_model == "Random Forest":
            st.image("assets/RF_evaluation.png", use_container_width=True)
            st.markdown("""
**Random Forest** prezentuje stabilne wyniki oraz umożliwia analizę ważności cech, co wspiera interpretację predykcji.
            """)
            st.markdown("#### Ważność cech - Random Forest")
            st.image("assets/RF_feature_importance.png", use_container_width=True)
        elif selected_model == "KNN":
            st.image("assets/KNN_evaluation.png", use_container_width=True)
            st.markdown("""
**K-Nearest Neighbors (KNN)** jest prosty w interpretacji, jednak jego skuteczność może być ograniczona przy dużych zbiorach danych.
            """)
        elif selected_model == "Decision Tree":
            st.image("assets/DT_evaluation.png", use_container_width=True)
            st.markdown("""
**Drzewo Decyzyjne** wyróżnia się przejrzystą strukturą, co ułatwia interpretację, choć osiąga niższe metryki.
            """)
            st.markdown("#### Ważność cech - Decision Tree")
            st.image("assets/DC_feature_importance.png", use_container_width=True)
            st.markdown("""
Widać, że cechy takie jak ST_Slope, Dławica wysiłkowa oraz Oldpeak mają największy wpływ na decyzję drzewa.
            """)
            st.markdown("#### Struktura drzewa decyzyjnego")
            st.image("assets/decision_tree.jpg", use_container_width=True)
            st.markdown("""
Drzewo pokazuje, jak kolejne warunki decyzyjne prowadzą do ostatecznego podziału na klasy (Choroba serca / Brak choroby).
            """)

    st.markdown("---")
    st.markdown("""
    ### Komentarz
    - **Stacking Classifier** i **Voting Classifier Soft** uzyskały najwyższe metryki (AUC, F1-score), co świadczy o skuteczności łączenia wyników wielu modeli.
    - **Random Forest**, **SVM** oraz **Logistic Regression** prezentują stabilne wyniki.
    - **Decision Tree** i **KNN** zwarcie charakteryzują się łatwością interpretacji i prostotą implementacji, mimo nieco niższych metryk.
    - W praktycznych zastosowaniach medycznych istotna jest nie tylko wysoka skuteczność (AUC), ale także przejrzystość interpretacji (Precision, Recall, F1-score).
    """)


# =============================================================================
#  ZAKŁADKA STRONY Z DOKUMENTACJĄ PROJEKTU
# =============================================================================
def page_about():
    st.title("📚 Dokumentacja projektu: CardioPredict")

    st.markdown("""
    ## **1. Wprowadzenie i Cel**

    Choroby układu krążenia są jedną z najczęstszych przyczyn zgonów na świecie. Wczesne wykrywanie symptomów i odpowiednia profilaktyka stanowią klucz do redukcji ryzyka i poprawy jakości życia pacjentów. Projekt **CardioPredict** ma na celu wspomaganie diagnostyki chorób serca za pomocą **uczenia maszynowego (ML)**, łącząc **analizę danych**, **trening modeli** oraz **czytelny interfejs** w technologii Streamlit.

    ### **Założenia projektu**  
    1. **Zastosowanie uczenia maszynowego** do identyfikacji i oceny czynników ryzyka chorób serca.  
    2. **Udostępnienie intuicyjnego interfejsu** pozwalającego osobom nietechnicznym korzystać z narzędzia.  
    3. **Zapewnienie interpretowalności** predykcji dzięki narzędziu SHAP Explainer.  
    4. **Możliwość przetwarzania wielu przypadków** (funkcjonalność masowej predykcji CSV).

    ---

    ## **2. Zbiór Danych**

    **Źródło:** *Kaggle* – Heart Failure Prediction Dataset, autor *FEDESORIANO* (wrzesień 2021).  
    Zbiór powstał z **połączenia 5 niezależnych źródeł** danych o chorobach serca:

    1. Cleveland (1990) – 303 obserwacje  
    2. Hungarian (1990) – 294 obserwacje  
    3. Switzerland (1989) – 123 obserwacje  
    4. Long Beach VA (1989) – 200 obserwacji  
    5. Stalog (Heart) Data Set (1990) – 270 obserwacji  

    **Razem:** 1190 rekordów, z czego 272 duplikaty. Ostatecznie zachowano **918** unikalnych obserwacji.

    ---

    ## **3. Uczenie Maszynowe w Projekcie**

    1. **Przygotowanie danych**:  
       - Walidacja odstających wartości,  
       - Skalowanie atrybutów ciągłych,  
       - Kodowanie zmiennych kategorycznych.

    2. **Trening i optymalizacja**:  
       - **Modele testowane:** Logistic Regression, SVM, Decision Tree, Random Forest, KNN, Voting i Stacking.  
       - **Optymalizacja hiperparametrów:**.  
       - **Walidacja krzyżowa** – wybór najlepszych ustawień na podstawie Accuracy, F1-score, Recall, Precision, AUC-ROC.

    3. **Wyjaśnialność (Explainable AI)**:  
       - **SHAP Explainer** pozwala zobaczyć, jak każda cecha wpływa na wynik modelu.  
       - Wykresy typu SHAP Waterfall zapewnia czytelność.

    ---

    ## **4. Aplikacja (Streamlit)**

    ### **Główne Zakładki**

    1. **Strona Główna** – Wprowadzenie do aplikacji.
    2. **Predykcja** – Formularz do oceny ryzyka choroby serca.
    3. **Import CSV** – Wczytywanie pliku z wieloma przypadkami.
    4. **Wizualizacje** – Porównanie pacjenta z populacją za pomocą wykresów radarowych, histogramów i boxplotów.
    5. **Analiza wyników** – Podsumowanie wartości pacjenta w odniesieniu do norm.
    6. **Skuteczność Predykcji** – Wyniki testowania modeli ML.
    7. **Dokumentacja projektu** – Opis techniczny i koncepcyjny aplikacji.

    ---

    ## **5. Podsumowanie**

    **CardioPredict** to projekt łączący zalety uczenia maszynowego i przyjaznego interfejsu:

    - **Analiza i przygotowanie danych** (oczyszczanie, skalowanie, kodowanie),  
    - **Wiele modeli** przetestowanych i zoptymalizowanych,  
    - **Interfejs użytkownika** (Streamlit) zapewniający intuicyjną obsługę,  
    - **Interpretacja wyników** (SHAP) zapewniająca przejrzystość działania modeli,  
    - **Wizualizacje** (radar, histogramy, kołowe, itp.) ułatwiające zrozumienie pozycji pacjenta na tle populacji.  

    Aplikacja może pełnić funkcję **wsparcia wstępnej diagnozy** oraz służyć celom edukacyjnym, demonstrując praktyczne zastosowanie ML w medycynie.
    """)




# =============================================================================
# START APLIKACJI
# =============================================================================
def main():
    st.set_page_config(
        page_title="CardioPredict",
        page_icon="❤️",
        layout="wide"
    )
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    inputs = sidebar_inputs()

    # Definiujemy zakładki
    tabs = st.tabs([
        "📌 Strona Główna",
        "🩺 Predykcja",
        "📂 Import CSV",
        "📊 Wizualizacje",
        "📈 Analiza wyników",
        "🎯 Skuteczność predykcji",
        "📚 Dokumentacja projektu"
    ])

    with tabs[0]:
        page_home(inputs)
    with tabs[1]:
        page_prediction(inputs)
    with tabs[2]:
        page_mass_prediction()
    with tabs[3]:
        page_visualizations(inputs)
    with tabs[4]:
        page_analysis(inputs)
    with tabs[5]:
        page_model_evaluation()
    with tabs[6]:
        page_about()

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # 2. Dodanie własnego, „przyklejonego” footera na dole ekranu
    custom_footer = """
    <div style="
        position:fixed;
        left:0;
        bottom:0;
        width:100%;
        background-color:#ffffff;
        color:#000000;
        text-align:center;
        padding:10px 0;
        font-size:1em;">
        Autor: Grzegorz Dróżdż | Kontakt: grzegorz.drozdz@edu.uekat.pl
    </div>
    """

    st.markdown(custom_footer , unsafe_allow_html=True)


if __name__ == '__main__':
    main()