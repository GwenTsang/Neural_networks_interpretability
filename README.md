# Neural_networks_interpretability : détection des parenthèses dans un LSTM bilingue

Ce dépôt contient le code source et les expériences pour l'article *« Détection des parenthèses dans un LSTM bilingue : Neurone partagé ou mécanisme distribué ? »*.

## Structure du projet

```
.
├── LSTM_models/
│   ├── best_lstm_model.pt          # Modèle entraîné (R² = 0.6971, neurone 250)
│   ├── LSTM_perf69.pt              # Checkpoint alternatif
│   └── readme.md
│
├── experiments/
│   ├── score_0_6971.ipynb                     		 # Entraînement du modèle principal
│   ├── interpretable_LSTM.ipynb  						# Expérience préliminaire
│   ├── interpretability_optimization_2layers.py        # Recherche hyperparamètres (2 couches)
│   │
│   ├── ablation/
│   │   └── notebooks/
│   │       ├── ablation.ipynb              # Expériences d'ablation (lésion, clampage) + expériences de profondeur
│   │       └── neuron_250_analysis.ipynb   # Analyse des délimiteurs (parenthèses vs crochets)
│   │
│   └── heatmaps/
│       ├── heatmaps.ipynb                  # Génération des heatmaps
│       ├── heatmap_baseline.png
│       ├── heatmap_clamp0.png
│       └── heatmap_clampK.png
│
└── training_corpus/
    ├── training_corpus_constitution.ipynb  # Script de préparation du corpus
    ├── reconstructed_lstm_corpus.txt       # Corpus filtré (3.6M caractères)
    └── reconstructed_training_corpus.txt
```

## Reproduction des résultats

### 1. Entraînement du modèle

```bash
# Ouvrir dans Google Colab (GPU recommandé)
experiments/score_0_6971.ipynb
```

Le notebook télécharge automatiquement les corpus depuis GitHub et entraîne le modèle pendant 40 époques.

**Hyperparamètres clés :**
- Embedding : 128 dimensions
- Hidden size : 512 neurones
- 1 couche LSTM
- Learning rate : 8.8 × 10⁻⁴
- Dropout : 0.17

### 2. Expériences d'ablation

```bash
# Nécessite best_lstm_model.pt dans le même dossier
experiments/ablation/notebooks/ablation.ipynb
```

**Conditions testées :**
- Baseline (sans intervention)
- Lésion : $c_{t,250} = 0$
- Clamp=0 : forcer "outside"
- Clamp=K : forcer "inside" (K ≈ 1.89)

### 3. Génération des heatmaps

```bash
experiments/heatmaps/heatmaps.ipynb
```

## Résultats principaux

### Probing (régression Ridge univariée)

| Condition | R² local | R² distribué |
|-----------|----------|--------------|
| Baseline | 0.4744 | 0.6674 |
| Lésion (=0) | -0.0001 | 0.5855 |
| Clamp=0 | -0.0001 | 0.5855 |
| Clamp=K | -0.0001 | 0.5788 |

### Profondeur d'imbrication

Le neurone 250 fonctionne comme un **détecteur binaire** (inside/outside), pas comme un compteur de profondeur :

| Chaîne | depth=1 | depth=2 | depth=3 | depth=4 |
|--------|---------|---------|---------|---------|
| `(def)` | 1.334 | — | — | — |
| `((def))` | 0.490 | 1.381 | — | — |
| `(((def)))` | 0.491 | 0.501 | 1.350 | — |
| `((((def))))` | 0.491 | 0.501 | 0.500 | 1.310 |

Corrélation activation vs profondeur : r = 0.19 (non significatif)

## Dépendances

```
torch >= 2.0
numpy
matplotlib
scikit-learn
```

## Auteurs

- Gwendal Tsang gwendal.tsang@gmail.com
- Anastasiia Belosevich a.belosevich@gmail.com

Université Sorbonne Nouvelle, 2026
