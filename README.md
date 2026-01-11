# Projet Deep Learning — Classification du risque de cancer du sein (Low / Medium / High)

**Étudiante :** GOKSEN Betul  
**Notebook :** `cancer_risk_classification_notebook.ipynb`  
**Dataset :** `cancer-risk-factors.csv`

---

## 1) Présentation du projet
Dans le cadre de ce DM, j’ai travaillé sur un cas d’usage inspiré d’un contexte réel de **cabinet médical**.  
L’objectif est d’aider les médecins à ne plus se baser uniquement sur l’âge pour recommander un dépistage, mais d’utiliser une approche **multifactorielle** (facteurs de santé, mode de vie, antécédents, etc.).

Le modèle doit classer les patientes selon trois niveaux de risque :
- **Low**
- **Medium**
- **High**

---

## 2) Priorité : sécurité avant accuracy
Dans le médical, l’erreur la plus critique est de **ne pas détecter une patiente “High Risk”** (faux négatif).  
C’est pourquoi l’accuracy globale n’est pas l’indicateur principal ici.

 **Objectif principal du projet : maximiser le Recall sur la classe “High” (viser 100%)**, quitte à générer davantage de faux positifs (sur-dépistage).

---

## 3) Démarche (cycle de projet type CRISP-DM)
### 3.1 Préparation des données
- **Filtrage** : conservation uniquement des cas liés au **cancer du sein** pour être cohérent avec la demande.
- **Nettoyage** : suppression des doublons et retrait des colonnes non pertinentes (ex. `Patient_ID`).
- **Normalisation** : standardisation des variables (via `StandardScaler`) afin de mettre des features comme l’âge, l’IMC, etc. sur la même échelle.

### 3.2 Gestion du déséquilibre des classes (défi principal)
Le dataset est fortement déséquilibré (très peu de cas **High**).

- Le premier modèle (ANN classique) est pénalisé par ce déséquilibre et détecte mal la classe “High”.
- J’ai donc testé des approches de rééquilibrage :
  - **SMOTE**
  - **SMOTE-Tomek**
Ces techniques sont appliquées uniquement sur l’entraînement, afin de préserver un test set réaliste.

### 3.3 Choix du modèle et optimisation
Après comparaison, le modèle le plus robuste et stable est :
- **Balanced Random Forest**

J’ai ensuite ajouté un point clé du projet :
- **Threshold tuning** : ajustement manuel du seuil de décision pour forcer une détection maximale des cas “High”.
Objectif : **0 faux négatif sur High**, même si cela augmente légèrement les fausses alertes.

### 3.4 Explicabilité (SHAP)
Un médecin doit pouvoir comprendre la décision : un modèle “boîte noire” est difficilement utilisable.

J’ai intégré **SHAP** pour :
- analyser l’importance globale des variables,
- expliquer **au cas par cas** pourquoi une patiente est classée “High”.

---

## 4) Résultats (synthèse)
-  **Recall (High Risk) : 100%** (aucune patiente à haut risque n’est oubliée)
-  **F1-score global : ~0.95** (selon configuration/seed)
-  **Interprétabilité** : décisions justifiées via SHAP (global + individuel)

> Remarque : les performances peuvent légèrement varier selon la séparation train/test et les paramètres, mais l’objectif prioritaire reste le Recall High.

---

## 5) Fichiers générés par le notebook
Le notebook sauvegarde des figures au format `.png` et crée un fichier `figures.zip` contenant toutes les images.

Exemples de figures générées (noms conservés tels quels) :
- `class_distribution.png`
- `correlation_matrix.png`
- `confusion_matrix_initial.png`
- `confusion_matrix_class_weight.png`
- `confusion_matrix_final.png`
- `threshold_analysis.png`
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`
- `shap_force_high_risk.png`
- `shap_waterfall_high_risk.png`
- `model_comparison.png`
- `figures.zip`

---

## 6) Exécution du projet
### Prérequis
- Python 3.9+ recommandé
- Jupyter Notebook / JupyterLab

### Installation des dépendances
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow shap
