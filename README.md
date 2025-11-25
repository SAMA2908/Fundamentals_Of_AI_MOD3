Il progetto analizza match ATP con Reti Bayesiane, confrontando l'impatto di Status (Diff. Ranking) e Performance (Differenze in Ace, Doppi Falli e Punti Vinti al Servizio) sul Risultato. Studia le vittorie a sorpresa, misurando l'importanza delle metriche di servizio in base alla Superficie.


Questo progetto utilizza le Reti Bayesiane per analizzare i match di tennis ATP (dal 2014 al 2024). L'obiettivo è comprendere le dinamiche causali che portano a un risultato, con un focus particolare sulle "vittorie a sorpresa" (Upset).

l modello confronta l'impatto di due macro-fattori sul risultato finale:

Status (Ex-Ante): Differenza di Ranking, Età, Altezza (informazioni note prima del match).

Performance (Ex-Post): Statistiche di gioco effettive come Ace, Doppi Falli e Punti Vinti al Servizio.

Inoltre, viene misurata l'influenza del contesto (Superficie) 


Per analisi, eseguire i file nel seguente ordine logico 

1. data_preparation.ipynb
Cosa fa: Carica i dataset grezzi (atp_matches_*.csv), gestisce i dati mancanti, calcola le nuove feature (es. Fav_On_Worst_Surface, Fav_Recent_Form) e discretizza le variabili numeriche.

Metodologia: Applica uno split temporale rigoroso:

Training: 2014-2023

Test: 2024 (i bin di discretizzazione vengono calcolati sul Train e applicati al Test per evitare Data Leakage).

Output: Genera i file puliti in processed_data/.

2. exploratory_analysis.ipynb (EDA)
Cosa fa: Analisi visiva dei dati.

Obiettivo: Verificare la qualità del dataset, analizzare la distribuzione delle vittorie e controllare il "Data Drift" (coerenza statistica) tra gli anni di training e l'anno di test (2024).

3. bayesian_models.ipynb
Cosa fa: Costruisce, addestra e valuta diversi modelli di Rete Bayesiana utilizzando la libreria pgmpy.

Modelli confrontati:

Expert Model: Struttura definita manualmente basata sulla conoscenza del tennis.

Naive Bayes: Baseline statistica.

Learned Model: Struttura appresa automaticamente dai dati (Hill Climbing Search).

Output: Accuratezza sul test set 2024 e analisi di scenari condizionali (es. "Come cambia la probabilità di vittoria se il favorito gioca sulla sua superficie peggiore?").