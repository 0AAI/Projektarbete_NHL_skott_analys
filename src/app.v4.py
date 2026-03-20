import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import os
import joblib 

# Jag ställer in sidans titel och layout
st.set_page_config(page_title="NHL Shot Analysis 2024", layout="wide")
FILE_PATH = r"C:\Projects\Projektarbete\data\shots_2024.csv"
MODEL_PATH = r"C:\Projects\Projektarbete\models\shot_models.pkl"

# Jag använder cache för att datan ska laddas snabbt utan att läsas in på nytt vid varje klick
@st.cache_data
def load_data():
    # Kollar om filen finns
    if not os.path.exists(FILE_PATH):
        return None
    
    # Läser in csv-filen
    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # Omvandlar avståndet till meter direkt här så det finns redo för alla grafer
    df['DistanceMeters'] = df['arenaAdjustedShotDistance'] * 0.3048
    
    # Rensar bort rader som saknar lagkod
    if 'teamCode' in df.columns:
        df = df[df['teamCode'].notna()]
        
    return df

# Här laddar jag in AI-modellerna som jag tränade i den andra filen
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

# Min funktion för att rita rinken i bakgrunden
def draw_rink(ax):
    # Ritar isen
    ax.add_patch(patches.Rectangle((0, -42.5), 100, 85, color="#f0f0f0", zorder=0))
    # Mållinjen och blålinjen
    ax.axvline(89, color="red", linewidth=2, zorder=1)
    ax.axvline(25, color="blue", linewidth=5, alpha=0.2, zorder=1)
    # Målgården
    ax.add_patch(patches.Rectangle((89, -3), 4, 6, color="lightblue", alpha=0.5, zorder=1))
    # Sargen
    ax.add_patch(patches.Rectangle((0, -42.5), 100, 85, fill=False, edgecolor="black", linewidth=2, zorder=5))
    
    # Tekningscirklarna
    for y_pos in [-22, 22]:
        ax.add_patch(patches.Circle((69, y_pos), 15, fill=False, edgecolor="red", linewidth=2, alpha=0.1))

# Här bygger jag själva applikationen
def main():
    st.title("NHL Shot Analysis 2024")
    st.markdown("Interaktiv analys av skott, mål och effektivitet.")

    # Hämtar datan
    df = load_data()
    # Hämtar modellerna
    models_data = load_models()
    
    if df is None:
        st.error(f"Kunde inte hitta filen på: {FILE_PATH}")
        return

    # Menyn i sidofältet
    analysis_type = st.sidebar.radio(
        "Välj Visualisering", 
        [
            "Översikt (KPI)", 
            "Heatmap: Målchans (%)",    
            "Heatmap: Returer", 
            "Stapel: Skott-typer",
            "Stapel: Avstånd (Meter)",  
            "AI Prediction: xG Model"
        ]
    )

    # Översiktssidan
    if analysis_type == "Översikt (KPI)":
        st.subheader("Säsongsöversikt")
        
        total_shots = len(df)
        total_goals = df['goal'].sum()
        shooting_pct = (total_goals / total_shots) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Totalt antal skott", f"{total_shots:,}")
        col2.metric("Totalt antal mål", f"{total_goals:,}")
        col3.metric("Målprocent (Snitt)", f"{shooting_pct:.2f}%")
        
        st.write("Exempel på data (Första 10 raderna):")
        st.dataframe(df.head(10))

    
    # Heatmapp: Visar den ANTAL mål (Volym) istället för procent
    elif analysis_type == "Heatmap: Målchans (%)":
        st.subheader("Var görs flest mål? (Volym)")
        st.markdown("Kartan visar **totalt antal mål** som gjorts från varje position under säsongen. En mörkare färg och högre siffra betyder att **fler mål** har gjorts därifrån.")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_rink(ax)
        
        # Här använder vi nu np.sum för att RÄKNA målen istället för att ta snittet
        hb = ax.hexbin(
            df['xCordAdjusted'], 
            df['yCordAdjusted'], 
            C=df['goal'], 
            gridsize=25, 
            reduce_C_function=np.sum,  # VIKTIGT: Summerar antalet mål (1+1+1...)
            cmap='Reds',               # Använder röd skala för mål
            edgecolors='gray', 
            linewidths=0.5,
            mincnt=1,                  # Visar rutan så fort minst 1 mål gjorts
            zorder=2
        )

        centers = hb.get_offsets()
        values = hb.get_array()
        max_val = values.max()

        for i in range(len(centers)):
            val = values[i]
            if val > 0:
                text_color = 'white' if val > max_val * 0.5 else 'black'
                if val >= 2:
                    ax.text(centers[i][0], centers[i][1], f'{int(val)}', 
                            ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')

        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Antal gjorda mål')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        
        st.pyplot(fig)

    # Heatmap för returer (Volym/Antal mål)
    elif analysis_type == "Heatmap: Returer":
        st.subheader("Var görs returmålen? (The Kill Zone)")
        st.markdown("Kartan visar **totalt antal mål** som gjorts på returer. Det är här du ska befinna dig för att slå in returer.")
        
        rebounds = df[df['shotRebound'] == 1]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_rink(ax)
        
        hb = ax.hexbin(
            rebounds['xCordAdjusted'], rebounds['yCordAdjusted'], 
            C=rebounds['goal'], 
            reduce_C_function=np.sum, 
            gridsize=20, 
            cmap='Reds', 
            mincnt=1,                
            edgecolors='gray', 
            linewidths=0.5, 
            zorder=2
        )
        
        centers = hb.get_offsets()
        values = hb.get_array()
        max_val = values.max() 

        for i in range(len(centers)):
            val = values[i]
            if val >= 1:
                text_color = 'white' if val > max_val * 0.5 else 'black'
                ax.text(centers[i][0], centers[i][1], f'{int(val)}', 
                        ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')

        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Antal returmål')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        
        st.pyplot(fig)

    # Stapeldiagram för skott-typer
    elif analysis_type == "Stapel: Skott-typer":
        st.subheader("Effektivitet per Skott-typ")
        
        shots_per_type = df.groupby('shotType').size().reset_index(name='TotalShots')
        goals_per_type = df[df['goal'] == 1].groupby('shotType').size().reset_index(name='Goals')
        
        stats = pd.merge(shots_per_type, goals_per_type, on='shotType', how='left').fillna(0)
        stats = stats[stats['TotalShots'] > 500]
        stats['Effektivitet'] = (stats['Goals'] / stats['TotalShots']) * 100
        stats = stats.sort_values('Effektivitet', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='shotType', y='Effektivitet', data=stats, palette='viridis', ax=ax)
        
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height() + 0.1, 
                    f'{p.get_height():.1f}%', ha="center", fontweight='bold')
        
        ax.set_ylabel("Målprocent (%)")
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)

        st.markdown("""
        **Förklaring av skott-typer:**
        * **DEFL** (Deflection / Styrning - Passiv) – 10.0%
        * **SNAP** (Snap Shot / "Snärtskott") – 8.5%
        * **BACK** (Backhand) – 8.4%
        * **TIP** (Tip-In / Styrning - Aktiv) – 7.8%
        * **WRIST** (Wrist Shot / Handledsskott) – 6.4%
        * **WRAP** (Wrap-around / "Köksvägen") – 5.2%
        * **SLAP** (Slap Shot / Slagskott) – 4.7%
        """)

    # Flik: Avstånd (Visar både Effektivitet och Volym)
    elif analysis_type == "Stapel: Avstånd (Meter)":
        
        # --- GRAF 1: EFFEKTIVITET (Sannolikhet) ---
        st.subheader("1. Sannolikhet: Hur stor chans är det att göra mål?")
        st.markdown("Grafen visar målprocenten. T.ex: *'Om du skjuter härifrån går 14% av skotten in'.*")
        
        bins_meter = [0, 3, 6, 9, 12, 15, 18]
        labels_meter = ['0-3m', '3-6m', '6-9m', '9-12m', '12-15m', '15-18m']
        df['DistanceGroupMeter'] = pd.cut(df['DistanceMeters'], bins=bins_meter, labels=labels_meter)
        
        eff_stats = df.groupby('DistanceGroupMeter', observed=True)['goal'].mean() * 100
        eff_stats = eff_stats.reset_index(name='Effektivitet')
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='DistanceGroupMeter', y='Effektivitet', data=eff_stats, palette='coolwarm', ax=ax1)

        for p in ax1.patches:
            if p.get_height() > 0:
                ax1.text(p.get_x() + p.get_width()/2., p.get_height() + 0.1, 
                        f'{p.get_height():.1f}%', ha="center", fontweight='bold')

        ax1.set_ylabel("Målchans (%)")
        ax1.set_xlabel("") 
        ax1.grid(axis='y', alpha=0.3)
        st.pyplot(fig1)

        st.markdown("---") 

        # --- GRAF 2: VOLYM (Varifrån kommer målen?) ---
        st.subheader("2. Volym: Varifrån görs flest mål?")
        st.markdown("Grafen visar hur stor andel av **alla ligans mål** som kommer från detta avstånd.")

        total_goals_in_league = df['goal'].sum()
        vol_stats = df.groupby('DistanceGroupMeter', observed=True)['goal'].sum().reset_index(name='AntalMål')
        vol_stats['AndelAvAllaMål'] = (vol_stats['AntalMål'] / total_goals_in_league) * 100

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='DistanceGroupMeter', y='AntalMål', data=vol_stats, palette='Greens', ax=ax2)

        for i, p in enumerate(ax2.patches):
            if p.get_height() > 0:
                antal = int(p.get_height())
                andel = vol_stats.loc[i, 'AndelAvAllaMål']
                ax2.text(p.get_x() + p.get_width()/2., p.get_height() + 5, 
                        f'{antal} st\n({andel:.1f}%)', ha="center", fontweight='bold')

        ax2.set_ylabel("Antal gjorda mål")
        ax2.set_xlabel("Avstånd från målet (Meter)")
        ax2.grid(axis='y', alpha=0.3)
        st.pyplot(fig2)

    # AI Prediction model 
    elif analysis_type == "AI Prediction: xG Model":
        st.subheader("End-to-End Machine Learning: xG Calculator")
        st.markdown("""
        Här använder vi en **Machine Learning-modell** (Random Forest) som har tränats på säsongens alla skott.
        Justera reglagen för att se hur målchansen förändras baserat på position.
        """)

        if models_data is None:
            st.warning("Du måste köra 'train_model.py' först för att träna AI:n!")
            return

        # INPUT och VISUALISERING (Överst) 
        col_in1, col_in2 = st.columns([1, 2]) # Smal kolumn för reglage, bred för karta
        
        with col_in1:
            st.markdown("**Ställ in position**")
            input_dist = st.slider("Avstånd till mål (meter)", 0.0, 30.0, 5.0)
            input_angle = st.slider("Vinkel (grader)", 0, 90, 0)
            st.caption("0 grader = Rakt framifrån. 90 grader = Död vinkel.")
            
            # Beräkna xG knapp och resultat
            st.markdown("---")
            if st.button("Beräkna Målchans (xG)"):
                model = models_data['model_rf']
                input_data = np.array([[input_dist, input_angle]])
                prob = model.predict_proba(input_data)[0][1]
                
                st.metric(label="Sannolikhet för mål", value=f"{prob*100:.1f}%")
                
                if prob > 0.20:
                    st.success("Mycket bra målchans!")
                elif prob > 0.05:
                    st.warning("Hyfsad chans.")
                else:
                    st.error("Låg chans.")

        with col_in2:
            st.markdown("**Visuell Position på Rink**")
            
            # Skapa figur för rinken
            fig, ax = plt.subplots(figsize=(8, 5))
            draw_rink(ax)
            
            # BERÄKNA POSITION FÖR GRAFIK 
            # Målet är vid x=89, y=0.
            # Vi måste konvertera input (meter) till rinkens koordinater (fot).
            # 1 meter = 3.281 fot.
            dist_ft = input_dist * 3.28084
            angle_rad = np.radians(input_angle)
            
            # Beräkna koordinater
            # Vi ritar alltid på "positiva" sidan (y > 0) för enkelhets skull
            shot_x = 89 - (dist_ft * np.cos(angle_rad))
            shot_y = dist_ft * np.sin(angle_rad)
            
            # Rita skottlinjen (streckad)
            ax.plot([shot_x, 89], [shot_y, 0], 'k--', alpha=0.3, linewidth=1)
            
            # Rita "Pucken" (Symbol)
            ax.plot(shot_x, shot_y, marker='o', color='black', markersize=12, label="Puck")
            
            # Rita text vid pucken
            ax.text(shot_x, shot_y + 3, "Skytt", ha='center', fontsize=9, fontweight='bold')

            # Justera axlar så vi ser rätt del av zonen
            ax.set_xlim(0, 100)
            ax.set_ylim(-42.5, 42.5)
            ax.set_aspect('equal')
            ax.axis('off') # Stänger av axlarna för renare look
            
            st.pyplot(fig)

        st.markdown("---")
        
        # MODELL UTVÄRDERING
        with st.expander("Visa Modell-Utvärdering (Tekniska detaljer)", expanded=True):
            st.info("Modellerna utvärderades på 20% testdata som modellerna ej tidigare sett.")
            col_m1, col_m2 = st.columns(2)
            
            metrics_lr = models_data['metrics_lr']
            metrics_rf = models_data['metrics_rf']
            
            with col_m1:
                st.write("**Modell A: Logistic Regression**")
                st.write(f"Accuracy: {metrics_lr['accuracy']:.2%}")
                st.write(f"AUC Score: {metrics_lr['auc']:.4f}")
            
            with col_m2:
                st.write("**Modell B: Random Forest (Vald)**")
                st.write(f"Accuracy: {metrics_rf['accuracy']:.2%}")
                st.write(f"AUC Score: {metrics_rf['auc']:.4f}")
                st.caption("Random Forest valdes för att den presterade likvärdigt men fångar icke-linjära samband bättre.")

if __name__ == "__main__":
    main()