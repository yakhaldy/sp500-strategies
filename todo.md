# TODO — SP500 ML Strategy

## État actuel (bugs corrigés)

Tous les scripts ont été corrigés et le pipeline complet tourne de bout en bout (preuves : voir `results/`). Bugs trouvés et corrigés :

- **`venv/` cassé** : le venv committé pointait vers un interpréteur d'une autre machine (`/home/yakhaldy/...`). Recréé + `requirements.txt` renseigné (pandas, numpy, scikit-learn, lightgbm, matplotlib, ta, joblib). Sur macOS, `lightgbm` a aussi besoin de `libomp` (`brew install libomp`) sinon `OSError: Library not loaded: @rpath/libomp.dylib`.
- **`scripts/gridsearch.py`** : le `return` de `grid_search_cv()` était indenté à l'intérieur de la boucle `for mname in model_name`, donc la fonction s'arrêtait après le 1er modèle testé (jamais de vraie comparaison). La fonction comparait aussi 3 pipelines incompatibles entre eux (steps `clf` pour rf/gb/lr) alors que le reste du code (`model_selection.py`, `create_signal.py`) attendait un pipeline avec un step nommé `model` (LightGBM) — deux architectures de pipeline différentes cohabitaient. Remplacé par **un seul pipeline canonique** (`build_pipeline()`, Imputer→Scaler→LGBMClassifier) partagé par tous les scripts, et une vraie recherche par grille (`ParameterGrid`) sur ses hyperparamètres. Renommé `time_series_split()` → `build_cv_splits()` (le nom que `create_signal.py` essayait déjà d'importer).
- **`scripts/model_selection.py`** : utilisait une variable `pipeline` jamais définie (le bloc qui la créait était commenté) → `NameError` immédiat. Réécrit pour utiliser `build_pipeline()`, et ajouté le plot `metric_train.png` (AUC train/val par fold) qui était un livrable demandé mais absent du code.
- **`scripts/create_signal.py`** : `from gridsearch import build_cv_splits` levait un `ImportError` (la fonction s'appelait `time_series_split`). Le signal ne couvrait en plus que la période train (walk-forward sur les folds) — rien n'était prédit sur la période test, alors que le backtest a besoin d'un signal sur train **et** test. Corrigé : le signal train reste du walk-forward (clone + fit par fold), et la période test est prédite par le pipeline déjà entraîné sur tout le train (`selected_model.pkl`), sans jamais toucher au test pendant l'entraînement.
- **`scripts/strategy.py`** : fichier vide (1 ligne). Le module de backtesting n'existait pas du tout. Implémenté : stratégie long-only binaire (signal > 0.5, poids normalisés à $1/jour), calcul du PnL avec la bonne règle anti-leakage (position du jour D × `return_d1_d2` de la ligne D, jamais re-décalée), benchmark SP500 ($1/jour investi), PnL cumulé + drawdown max + Sharpe/volatilité (bonus), sauvegarde de `strategy.png` et `results.csv`.
- **`results/strategy/report.md`** : rédigé (était vide).

Tous les livrables de la structure du repo existent maintenant et ont été régénérés par une exécution réelle (pas des fichiers stubs) :
`Time_series_split.png`, `ml_metrics_train.csv`, `metric_train.png`, `top_10_feature_importance.csv`, `selected_model.pkl`, `selected_model.txt`, `ml_signal.csv`, `strategy.png`, `results.csv`, `report.md`.

## Comment relancer chaque étape

Pré-requis une seule fois :
```bash
cd /Users/yahya/Desktop/sp500-strategies
brew install libomp   # requis par lightgbm sur macOS, si pas déjà fait
```
Toutes les commandes ci-dessous s'exécutent **depuis la racine du projet** (les scripts utilisent des chemins relatifs `data/...` et `results/...`) :

### Étape 1 — Cross-validation + sélection du modèle (~5 min)
```bash
./venv/bin/python3 scripts/model_selection.py
```
Charge `data/all_stocks_5yr.csv`, calcule les features (RSI/MACD/Bollinger), split train (<2017) / test (≥2017), construit les 10 folds Time Series Split, lance la grid search LightGBM, sélectionne le meilleur jeu d'hyperparamètres par AUC de validation moyenne, l'évalue sur le test set, puis sauvegarde :
- `results/cross-validation/Time_series_split.png`
- `results/cross-validation/ml_metrics_train.csv`
- `results/cross-validation/metric_train.png`
- `results/cross-validation/top_10_feature_importance.csv`
- `results/selected-model/selected_model.pkl`
- `results/selected-model/selected_model.txt`

### Étape 2 — Génération du signal ML (~25 s)
```bash
./venv/bin/python3 scripts/create_signal.py
```
Recharge `selected_model.pkl`, refait le walk-forward sur les 10 folds train (clone du pipeline, fit sur le train du fold, predict_proba sur la validation du fold), puis prédit la période test avec le pipeline déjà entraîné sur tout le train. Sauvegarde `results/selected-model/ml_signal.csv`.

### Étape 3 — Backtest de la stratégie (~10 s)
```bash
./venv/bin/python3 scripts/strategy.py
```
Convertit le signal en stratégie long-only binaire, calcule le PnL vs SP500, sauvegarde `results/strategy/strategy.png` et `results/strategy/results.csv`, affiche les métriques (PnL, max drawdown, volatilité, Sharpe) train/test dans le terminal.

## Ce qu'il reste à faire (au choix, non bloquant)

1. **Relire `results/strategy/report.md`** et vérifier que les chiffres correspondent bien à ta dernière exécution (les résultats sont déterministes — `random_state=42` partout — donc ils ne bougeront pas tant que le code/les données ne changent pas).
2. **(Optionnel) Élargir la grille d'hyperparamètres** dans `scripts/gridsearch.py::param_grid` si tu veux pousser l'AUC de validation au-delà de 0.5175 (actuellement 8 combinaisons testées). Attention : plus de combinaisons × 10 folds = plus de temps de calcul.
3. **(Optionnel) Tester une autre stratégie** que le long-only binaire (ex. long-short ternaire, ou poids proportionnels au signal) dans `scripts/strategy.py` — la structure du fichier permet d'ajouter une fonction `build_..._strategy()` alternative sans toucher au reste.
4. **(Optionnel) RNN** : l'énoncé le mentionne comme piste d'apprentissage facultative, non implémentée ici (aucun livrable ne l'exige).
5. **Nettoyer `r.txt`** à la racine (note perso avec un lien YouTube sur les Bollinger Bands) si tu n'en as plus besoin — pas lié au pipeline.
