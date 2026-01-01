# NHL API data loader och skott/mål-analys

import pandas as pd
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import os

# Din sökväg till filen (anpassad för Windows)
file_path = r"C:\Projects\Projektarbete\data\shots_2024.csv"

print(f"Letar efter filen här: {file_path}")

# 1. Kolla om filen faktiskt finns där
if os.path.exists(file_path):
    print("Filen hittades! Laddar in data... (Vänta lite)")

    # 2. Läs in filen
    # low_memory=False behövs för att MoneyPuck-filer är stora
    df = pd.read_csv(file_path, low_memory=False)

    # 3. Visa resultatet i terminalen
    print("\n--- SUCCÉ! Filen är inläst ---")
    print(f"Antal rader (skott): {len(df)}")
    print(f"Antal kolumner (variabler): {len(df.columns)}")

    print("\n--- SÅ HÄR SER DATAN UT (De 5 första raderna) ---")
    # Detta gör att vi ser alla kolumner i utskriften
    pd.set_option('display.max_columns', None) 
    print(df.head())

else:
    print("\nFEL: Datorn hittar inte filen.")
    print("Kolla att filnamnet i mappen verkligen är 'shots_2024.csv'")

# Sökväg till din fil
FILE_PATH = r"C:\Projects\Projektarbete\data\shots_2024.csv"

# Välj vad vi ska titta på: "ALLA", "POWERPLAY", "RETURER"
FILTER_TYPE = "ALLA" 
# =====================

def draw_rink(ax):
    """Ritar en halv hockeyrink (Anfallszonen)"""
    # Isen (Vit bakgrund)
    ax.add_patch(patches.Rectangle((0, -42.5), 100, 85, color="white", zorder=0))
    
    # Linjer
    ax.axvline(89, color="red", linestyle="-", linewidth=2, zorder=1) # Mållinje
    ax.axvline(25, color="blue", linewidth=5, alpha=0.2, zorder=1)    # Blålinje (ca)
    ax.axvline(0, color="red", linewidth=5, alpha=0.2, zorder=1)      # Mittlinje
    
    # Målgården (Ljusblå)
    ax.add_patch(patches.Rectangle((89, -3), 4, 6, color="lightblue", alpha=0.5, zorder=1))
    
    # Rinkens ram (Sargen)
    ax.add_patch(patches.Rectangle((0, -42.5), 100, 85, fill=False, edgecolor="black", linewidth=2, zorder=5))

    # Tekningscirklar (Röda)
    for y_pos in [-22, 22]:
        circle = patches.Circle((69, y_pos), 15, fill=False, edgecolor="red", linewidth=2, zorder=1)
        ax.add_patch(circle)
        dot = patches.Circle((69, y_pos), 1, color="red", zorder=1)
        ax.add_patch(dot)

def run_analysis():
    print(f"--- Läser in data från {FILE_PATH} ---")
    
    if not os.path.exists(FILE_PATH):
        print("FEL: Hittar inte filen.")
        return

    # Läs in datan
    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # Filtrera: Vi vill bara se MÅL (goal == 1)
    goals = df[df['goal'] == 1].copy()
    print(f"Hittade totalt {len(goals)} mål i datan.")

    # Applicera ditt valda filter
    if FILTER_TYPE == "POWERPLAY":
        goals = goals[goals['isPowerPlay'] == 1]
        title = "Heatmap: Mål i Powerplay"
    elif FILTER_TYPE == "RETURER":
        goals = goals[goals['shotGeneratedRebound'] == 1]
        title = "Heatmap: Mål på Returer"
    else:
        title = "Heatmap: Alla Mål"

    print(f"Visar graf för: {len(goals)} st mål...")

    # --- RITA GRAFEN ---
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # 1. Rita rinken
    draw_rink(ax)

    # 2. Skapa Heatmap
    # xCordAdjusted = Längdled (0-100), yCordAdjusted = Breddled (-42 till 42)
    sns.kdeplot(
        x=goals['xCordAdjusted'], 
        y=goals['yCordAdjusted'], 
        cmap="Reds",    # Färgskala (Röd)
        fill=True,      # Fyll färgen
        thresh=0.05,    # Ta bort de allra glesaste områdena
        alpha=0.7,      # Genomskinlighet
        zorder=2
    )

    # Inställningar
    plt.xlim(0, 100)
    plt.ylim(-42.5, 42.5)
    plt.title(title, fontsize=15)
    plt.xlabel("Avstånd (ft)")
    plt.ylabel("Bredd (ft)")
    plt.gca().set_aspect('equal') # Håll proportionerna på rinken
    
    print("Visar grafen nu...")
    plt.show()

if __name__ == "__main__":
    run_analysis()