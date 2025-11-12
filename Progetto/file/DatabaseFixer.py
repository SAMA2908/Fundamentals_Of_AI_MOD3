import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================================
# FASE 0: CONFIGURAZIONE E CARICAMENTO DATI (REVISITATO)
# ==============================================================================

# Definisce il percorso: risale da 'file/' e scende in 'data/'
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data'

START_YEAR = 2014
END_YEAR = 2024
years = range(START_YEAR, END_YEAR + 1)

#COLONNE STATISTICHE ESSENZIALI (Più COERENTI)
# Abbiamo sostituito 'unforced_errors' con statistiche di servizio più affidabili.
REQUIRED_COLS_STATS = [
    'w_ace', 'l_ace', 'w_df', 'l_df',         # Ace e Doppi Falli
    'w_svpt', 'l_svpt',                       # Punti al Servizio
    'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', # Efficacia 1a Palla
    'w_2ndWon', 'l_2ndWon',                   # Efficacia 2a Palla
    'w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced' # Palle Break
]
REQUIRED_COLS_RANK = ['winner_rank', 'loser_rank', 'surface']
REQUIRED_ALL = REQUIRED_COLS_STATS + REQUIRED_COLS_RANK

# Colonne totali da estrarre (comprese le chiavi identificative)
INITIAL_COLS = ['tourney_id', 'winner_id', 'loser_id'] + REQUIRED_ALL

# 1. Caricamento e Unione di tutti gli anni, cercando i file validi
matches_list = []
found_files_count = 0
skipped_files = []

print(f"--- Inizio Caricamento Dati ({START_YEAR}-{END_YEAR}) ---")
print(f"Cartella di ricerca: {DATA_PATH}")

# Cerca tutti i file che iniziano con 'atp_matches_' nella cartella DATA_PATH
all_match_files = sorted(list(DATA_PATH.glob('atp_matches_*.csv')))

if not all_match_files:
    raise FileNotFoundError(f"Nessun file 'atp_matches_*.csv' trovato in {DATA_PATH}. Controlla i nomi dei file.")

for match_file in all_match_files:
    try:
        # Usiamo low_memory=False per evitare avvisi di DtypeWarning
        df = pd.read_csv(match_file, low_memory=False)
        current_cols = df.columns.tolist()
        
        # Verifica quali colonne essenziali MANCANO
        missing_cols = [col for col in INITIAL_COLS if col not in current_cols]
        
        if not missing_cols:
            # CASO 1: File COMPLETO
            matches_list.append(df[INITIAL_COLS])
            found_files_count += 1
        else:
            # CASO 2: File INCOMPLETO -> CERCA DI RECUPERARE (Logica di salvataggio)
            
            # Controlla se le colonne chiave non statistiche sono presenti.
            critical_missing = [col for col in INITIAL_COLS if col not in current_cols and col not in REQUIRED_COLS_STATS]

            if critical_missing:
                # Se mancano colonne CRITICHE (ID, RANK, SURFACE), salta il file
                skipped_files.append(f"{match_file.name} (SALTO: Mancano colonne CRITICHE: {', '.join(critical_missing)})")
            else:
                # Aggiunge le colonne statistiche mancanti e riempie con NaN
                for col in missing_cols:
                    df[col] = np.nan
                
                # Ora che il DF ha tutte le colonne, possiamo selezionarle e aggiungerlo
                matches_list.append(df[INITIAL_COLS])
                found_files_count += 1
                
                # Registra l'operazione
                skipped_files.append(f"{match_file.name} (RECUPERO: Aggiunte e riempite con NaN le colonne: {', '.join(missing_cols)})")

    except Exception as e:
        skipped_files.append(f"{match_file.name} (ERRORE di lettura: {e})")

# Gestione dei file caricati
if not matches_list:
    raise FileNotFoundError("Nessun match valido è stato caricato. Interrompendo l'esecuzione.")

full_df = pd.concat(matches_list, ignore_index=True)

# 2. Pulizia finale: Rimuove le righe con valori mancanti
initial_rows = len(full_df)
# La pulizia è basata sul nuovo set di colonne REQUIRED_ALL
full_df.dropna(subset=REQUIRED_ALL, inplace=True)
dropped_rows = initial_rows - len(full_df)

print(f"\nSUCCESSO: Dataset Integrato e Pulito con {len(full_df)} partite complete.")
print(f"NOTA: Rimosse {dropped_rows} righe con statistiche mancanti (dopo l'unione).")

if skipped_files:
    print(f"\nATTENZIONE: Log di Caricamento/Recupero:")
    for sf in skipped_files:
        print(f"- {sf}")


# ------------------------------------------------------------------------------
# FASE 1: FEATURE ENGINEERING (Logica Favorito vs Sfavorevole) - AGGIORNATA
# ------------------------------------------------------------------------------

final_df = pd.DataFrame()

# Il favorito è il giocatore con il rank più basso (numero inferiore)
full_df['is_winner_favorite'] = full_df['winner_rank'] < full_df['loser_rank']

# --- A. Risultato del Favorito (Vittoria/Sconfitta) ---
final_df['Risultato_Favorevole'] = full_df['is_winner_favorite'].astype(int).map({1: 'Vittoria', 0: 'Sconfitta'})

# --- B. Nodo Contesto: Superficie ---
final_df['Superficie'] = full_df['surface'].astype(str)

# --- C. Nodo Status: Differenza di Ranking (Sfavorevole - Favorito) ---
final_df['Diff_Rank'] = full_df.apply(
    lambda row: row['loser_rank'] - row['winner_rank'] if row['is_winner_favorite'] else row['winner_rank'] - row['loser_rank'], 
    axis=1
)

# --- D. Nodi Performance: Nuove Differenze di Statistiche (Favorito - Sfavorevole) ---
# Creiamo delle differenze normalizzate (perché usiamo valori assoluti come 'svpt')
# o differenze semplici per le metriche più dirette (Ace, DF).

# Funzione helper per ottenere il valore del Favorito o dello Sfavorevole
def get_fav_sfav_val(row, stat_w, stat_l):
    """Calcola la differenza (Fav - Sfav) per una data statistica."""
    if row['is_winner_favorite']:
        # Vincitore è il Favorito: Fav - Sfav = Winner - Loser
        return row[stat_w] - row[stat_l]
    else:
        # Perdente è il Favorito: Fav - Sfav = Loser - Winner
        return row[stat_l] - row[stat_w]

# 1. Diff_Ace (Ace_Fav - Ace_Sfav)
final_df['Diff_Ace'] = full_df.apply(lambda row: get_fav_sfav_val(row, 'w_ace', 'l_ace'), axis=1)

# 2. Diff_DF (DF_Fav - DF_Sfav)
# Nota: un valore NEGATIVO in DF_Fav significa che il Favorito ha fatto MENO doppi falli.
final_df['Diff_DF'] = full_df.apply(lambda row: get_fav_sfav_val(row, 'w_df', 'l_df'), axis=1)

# 3. Percentuale Vinta Totale al Servizio (Service Points Won %)
# E' una metrica DERIVATA che è molto forte: (1stWon + 2ndWon) / SvPt
full_df['w_service_win_perc'] = (full_df['w_1stWon'] + full_df['w_2ndWon']) / full_df['w_svpt']
full_df['l_service_win_perc'] = (full_df['l_1stWon'] + full_df['l_2ndWon']) / full_df['l_svpt']

# Diff_ServiceWin (ServiceWin%_Fav - ServiceWin%_Sfav)
final_df['Diff_ServiceWinPerc'] = full_df.apply(
    lambda row: get_fav_sfav_val(row, 'w_service_win_perc', 'l_service_win_perc'), 
    axis=1
)
# Sostituiamo i possibili NaN/Inf che potrebbero essere creati da divisioni per zero
final_df['Diff_ServiceWinPerc'].replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(subset=['Diff_ServiceWinPerc'], inplace=True)


# DataFrame finale, con tutte le variabili numeriche e la superficie
# (Abbiamo rimosso 'Diff_ErroriNF' e aggiunto le nuove metriche)
data_numerica_pronta = final_df[['Superficie', 'Diff_Rank', 'Diff_Ace', 'Diff_DF', 'Diff_ServiceWinPerc', 'Risultato_Favorevole']].copy()

print("\n--- FASE 1 COMPLETATA: DataFrame Numerico Pronto ---")
print("Questo DataFrame contiene i dati puliti e le differenze Favorevole-Sfavorevole.")
print("\nEsempio di Dati (Numerici):")
print(data_numerica_pronta.head())

# ==============================================================================
# PROSSIMO STEP: DISCRETIZZAZIONE E MODELLAZIONE (FASE DI PROGETTO)
# ==============================================================================

# Il prossimo passo sarà riprendere questo DataFrame (data_numerica_pronta)
# e applicare la Discretizzazione (Passo 4) per preparare i dati
# per la Rete Bayesiana.