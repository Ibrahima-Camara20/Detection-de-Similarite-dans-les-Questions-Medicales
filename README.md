# Détection de similarité de questions médicales avec BioBERT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![BioBERT](https://img.shields.io/badge/Model-BioBERT-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## Vue d'ensemble
Ce projet détecte si deux questions de patients ont la **même intention médicale**, même si elles sont formulées différemment (ex : *"Quels sont les symptômes de la grippe ?"* et *"Comment savoir si j'ai attrapé la grippe ?"*).

Il repose sur le **transfer learning** : le modèle **BioBERT** (`dmis-lab/biobert-base-cased-v1.2`) est fine-tuné sur des paires de questions médicales. Le modèle est utilisé comme classifieur de paires (les deux questions sont lues ensemble, 2 classes : *différentes* / *similaires*).

### Fonctionnalités
* **Modèle spécialisé biomédical** : BioBERT, pré-entraîné sur PubMed.
* **Français et anglais** : les questions sont traduites vers l'anglais par un **modèle de traduction local** (`Helsinki-NLP/opus-mt-fr-en`). Aucun service externe, aucune limite de requêtes. Un texte déjà en anglais n'est pas traduit.
* **Résultats** : **83,77 % d'accuracy** et **0,84 de F1** sur le jeu de test (jeu équilibré 50/50, donc 50 % au hasard). Voir les limites plus bas.
* **Interface Streamlit** avec score de confiance et traduction utilisée par le modèle.

## Données
Jeu de données : [`medical_questions_pairs`](https://huggingface.co/datasets/curaihealth/medical_questions_pairs) : 3 048 paires de questions, 50 % similaires et 50 % différentes. Découpage 80 % entraînement / 20 % test (`random_state=42`).

## Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/Ibrahima-Camara20/D-tection-de-Similarit-dans-les-Questions-M-dicales.git
   cd D-tection-de-Similarit-dans-les-Questions-M-dicales
   ```

2. **Installer les dépendances :**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Le modèle** est téléchargé automatiquement au premier lancement depuis le Hugging Face Hub : [`Ibrahima20/bert-question-pairs-similarity`](https://huggingface.co/Ibrahima20/bert-question-pairs-similarity) (environ 400 Mo). Le modèle de traduction (environ 300 Mo) est aussi téléchargé une seule fois. Il n'y a **aucun dossier à télécharger ou renommer à la main**. Une connexion internet est nécessaire au premier lancement.

## Utilisation
```bash
streamlit run app.py
```
Ouvrir l'adresse affichée (par défaut http://localhost:8501), saisir deux questions et cliquer sur **Analyser la similarité**.

## Entraînement
Le notebook `Question_similaire.ipynb` (Google Colab, GPU T4) contient l'entraînement avec la classe `Trainer` de Hugging Face :

| Paramètre | Valeur |
|---|---|
| Époques | 3 |
| Taille de batch | 16 |
| Longueur maximale | 128 tokens |
| Autres paramètres | valeurs par défaut (taux d'apprentissage 5e-5, AdamW) |

Une expérience sur 10 époques montre du sur-apprentissage après l'époque 3 (la perte de test remonte). L'arrêt à 3 époques a été choisi à la main.

## Limites
* Le jeu de test sert aussi à choisir le nombre d'époques (pas de validation séparée) : le score est un peu optimiste.
* Une même question apparaît dans deux paires du jeu de données : un découpage aléatoire peut mettre la même question dans l'entraînement et dans le test.
* Un seul entraînement, sans comparaison avec une baseline simple (ex : TF-IDF + régression logistique).
* Le traducteur local ne gère que le sens français vers anglais.
* L'outil compare des questions. Il ne donne aucun conseil médical.

## Structure du dépôt
```
app.py                  # application Streamlit
requirements.txt        # dépendances
Question_similaire.zip  # notebook d'entraînement
Rapport_*.pdf           # rapport du projet
```
