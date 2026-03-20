import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# INSTÄLLNINGAR 
FILE_PATH = r"C:\Projects\Projektarbete\data\shots_2024.csv"
MODEL_DIR = r"C:\Projects\Projektarbete\models"

def train_models():
    print("--- 1. Laddar data... ---")
    if not os.path.exists(FILE_PATH):
        print("FEL: Hittar inte datafilen.")
        return

    # Ladda in nödvändiga kolumner
    # Vi behöver avstånd, vinkel och om det blev mål
    # MoneyPuck har ofta 'shotAngle' eller 'shotAngleAdjusted'. Vi kollar vad som finns.
    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # Förbered features (X) och target (y)
    # Vi konverterar fot till meter för konsekvensens skull
    df['distance_meter'] = df['arenaAdjustedShotDistance'] * 0.3048
    
    # Hantera vinkel (använd absolutbelopp för att vänster/höger sida inte ska spela roll)
    # Om 'shotAngleAdjusted' finns använder vi den, annars 'shotAngle'
    if 'shotAngleAdjusted' in df.columns:
        df['angle_abs'] = df['shotAngleAdjusted'].abs()
    else:
        df['angle_abs'] = df['shotAngle'].abs()

    # Välj ut datan vi ska träna på
    data = df[['distance_meter', 'angle_abs', 'goal']].dropna()
    
    X = data[['distance_meter', 'angle_abs']]
    y = data['goal']

    print(f"Data redo. Antal skott att träna på: {len(X)}")

    # 2. Dela upp i Träning (80%) och Test (20%) ---
    # Detta är viktigt för att kunna "Utvärdera" modellen (Krav 6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Modell A: Logistic Regression (Enkel) 
    print("\nTränar Modell A (Logistic Regression)...")
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)
    
    # Utvärdera A
    preds_lr = model_lr.predict(X_test)
    probs_lr = model_lr.predict_proba(X_test)[:, 1] # Sannolikheten
    acc_lr = accuracy_score(y_test, preds_lr)
    auc_lr = roc_auc_score(y_test, probs_lr)
    print(f"  -> Accuracy: {acc_lr:.4f}")
    print(f"  -> AUC Score: {auc_lr:.4f} (Högre är bättre)")

    # 4. Modell B: Random Forest (Avancerad)
    print("\nTränar Modell B (Random Forest)... detta kan ta nån minut...")
    # Vi begränsar djupet lite så filen inte blir gigantisk
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train)

    # Utvärdera B
    preds_rf = model_rf.predict(X_test)
    probs_rf = model_rf.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, preds_rf)
    auc_rf = roc_auc_score(y_test, probs_rf)
    print(f"  -> Accuracy: {acc_rf:.4f}")
    print(f"  -> AUC Score: {auc_rf:.4f}")

    # 5. Spara modellerna 
    print("\n--- Sparar modeller ---")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Vi sparar både modellen och resultaten i en "dictionary" så vi kan visa dem i appen
    results = {
        'model_lr': model_lr,
        'metrics_lr': {'accuracy': acc_lr, 'auc': auc_lr},
        'model_rf': model_rf,
        'metrics_rf': {'accuracy': acc_rf, 'auc': auc_rf}
    }

    save_path = os.path.join(MODEL_DIR, "shot_models.pkl")
    joblib.dump(results, save_path)
    print(f" Klart! Modeller sparade till: {save_path}")

if __name__ == "__main__":
    train_models()