import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import os

# --- KONFIGURATION ---
st.set_page_config(page_title="NHL Shot Analysis 2024", layout="wide")
FILE_PATH = r"C:\Projects\Projektarbete\data\shots_2024.csv"

# --- 1. LADDA DATA (Cachad för snabbhet) ---
@st.cache_data
def load_data():
    if not os.path.exists(FILE_PATH):
        return None
    
    # Läs in datan
    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # Skapa hjälpvariabler
    df['DistanceMeters'] = df['arenaAdjustedShotDistance'] * 0.3048
    
    # Filtrera bort "Ogiltiga" lag om det finns några
    if 'teamCode' in df.columns:
        df = df[df['teamCode'].notna()]
        
    return df

# --- 2. RITA RINKEN (Hjälpfunktion) ---
def draw_rink(ax):
    """Ritar rinken i bakgrunden"""
    ax.add_patch(patches.Rectangle((0, -42.5), 100, 85, color="#f0f0f0", zorder=0))
    ax.axvline(89, color="red", linewidth=2, zorder=1)
    ax.axvline(25, color="blue", linewidth=5, alpha=0.2, zorder=1)
    ax.add_patch(patches.Rectangle((89, -3), 4, 6, color="lightblue", alpha=0.5, zorder=1))
    ax.add_patch(patches.Rectangle((0, -42.5), 100, 85, fill=False, edgecolor="black", linewidth=2, zorder=5))
    
    # Tekningscirklar
    for y_pos in [-22, 22]:
        ax.add_patch(patches.Circle((69, y_pos), 15, fill=False, edgecolor="red", linewidth=2, alpha=0.1))

# --- 3. HUVUDPROGRAM ---
def main():
    st.title("🏒 NHL Shot Analysis 2024")
    st.markdown("Interaktiv analys av skott, mål och effektivitet.")

    # Ladda data
    df = load_data()
    
    if df is None:
        st.error(f"Kunde inte hitta filen på: {FILE_PATH}")
        return

    # --- SIDOFÄLT (SIDEBAR) ---
    st.sidebar.header("Filter")
    
    # Lag-väljare (Om man vill se specifikt lag)
    # Vi kollar både homeTeamCode och awayTeamCode för att hitta alla lag
    all_teams = sorted(list(set(df['homeTeamCode'].unique().tolist() + df['awayTeamCode'].unique().tolist())))
    selected_team = st.sidebar.selectbox("Välj Lag (eller Alla)", ["Alla"] + all_teams)
    
    # Filtrera datan om ett lag är valt
    if selected_team != "Alla":
        # Vi vill ha skott där det valda laget var det SKJUTANDE laget
        # I MoneyPuck heter skyttens lag ofta 'teamCode' eller så får vi gissa baserat på home/away
        # Enklast i denna data: Vi kollar kolumnen 'team' (som brukar vara HOME/AWAY) och matchar med koden.
        # Men för enkelhets skull i denna PoC, låt oss anta att vi filtrerar på hela datasetet först.
        # OBS: MoneyPuck har en kolumn 'eventTeamCode' eller liknande. 
        # Låt oss kolla om 'teamCode' finns (vanligt i bearbetad data), annars kör vi utan filter i version 1.
        pass # Vi kör på alla lag just nu för att inte krascha om kolumnen saknas

    # Välj analys-typ
    analysis_type = st.sidebar.radio(
        "Välj Visualisering", 
        ["Översikt (KPI)", "Heatmap: Alla Mål", "Heatmap: Returer", "Heatmap: One-Timers", "Stapel: Skott-typer"]
    )

    # --- VISA DATA ---
    
    # 1. ÖVERSIKT
    if analysis_type == "Översikt (KPI)":
        st.subheader("Säsongsöversikt")
        
        # KPI:er
        total_shots = len(df)
        total_goals = df['goal'].sum()
        shooting_pct = (total_goals / total_shots) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Totalt antal skott", f"{total_shots:,}")
        col2.metric("Totalt antal mål", f"{total_goals:,}")
        col3.metric("Målprocent (Snitt)", f"{shooting_pct:.2f}%")
        
        st.markdown("---")
        st.write("Exempel på data (Första 5 raderna):")
        st.dataframe(df.head())

    # 2. HEATMAP: ALLA MÅL
    elif analysis_type == "Heatmap: Alla Mål":
        st.subheader("Var görs målen?")
        
        goals = df[df['goal'] == 1]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_rink(ax)
        
        hb = ax.hexbin(
            goals['xCordAdjusted'], goals['yCordAdjusted'], 
            gridsize=25, cmap='Reds', mincnt=5, edgecolors='gray', linewidths=0.5
        )
        ax.set_title(f"Alla Mål ({len(goals)} st)")
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        
        st.pyplot(fig)

    # 3. HEATMAP: RETURER
    elif analysis_type == "Heatmap: Returer":
        st.subheader("The Kill Zone: Returer")
        st.markdown("Visar effektiviteten (Målprocent) på skott som är returer.")
        
        rebounds = df[df['shotRebound'] == 1]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_rink(ax)
        
        hb = ax.hexbin(
            rebounds['xCordAdjusted'], rebounds['yCordAdjusted'], 
            C=rebounds['goal'], reduce_C_function=np.mean,
            gridsize=20, cmap='Reds', mincnt=5, edgecolors='gray', linewidths=0.5, vmax=0.4
        )
        
        # Lägg till text
        centers = hb.get_offsets()
        values = hb.get_array()
        for i in range(len(centers)):
            val = values[i]
            if not np.isnan(val) and val > 0.01:
                 text_color = 'white' if val > 0.2 else 'black'
                 ax.text(centers[i][0], centers[i][1], f'{val*100:.0f}', 
                         ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')

        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Målprocent')
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        
        st.pyplot(fig)

    # 4. HEATMAP: ONE-TIMERS
    elif analysis_type == "Heatmap: One-Timers":
        st.subheader("One-Timers & Snabba avslut")
        st.markdown("Skott inom 3 sekunder från passning (ej returer). Visar **Antal Mål** (Volym).")
        
        quick_shots = df[
            (df['shotRebound'] == 0) & 
            (df['timeSinceLastEvent'] <= 3.0) & 
            (df['lastEventCategory'] != 'FAC') & 
            (df['shotType'].isin(['SLAP', 'SNAP', 'WRIST']))
        ]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_rink(ax)
        
        # Cirklar
        for y_pos in [-22, 22]:
            ax.add_patch(patches.Circle((69, y_pos), 15, fill=False, edgecolor="red", linewidth=2, alpha=0.3))

        hb = ax.hexbin(
            quick_shots['xCordAdjusted'], quick_shots['yCordAdjusted'], 
            C=quick_shots['goal'], reduce_C_function=np.sum,
            gridsize=25, cmap='Greens', mincnt=1, edgecolors='gray', linewidths=0.5
        )
        
        # Text
        centers = hb.get_offsets()
        values = hb.get_array()
        for i in range(len(centers)):
            val = values[i]
            if val >= 1:
                text_color = 'white' if val > values.max() * 0.5 else 'black'
                ax.text(centers[i][0], centers[i][1], f'{int(val)}', 
                        ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')

        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Antal Mål')
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal')
        
        st.pyplot(fig)

    # 5. STAPEL: SKOTT-TYPER
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

if __name__ == "__main__":
    main()