#  Détection de Similarité de Questions Médicales avec BioBERT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![BioBERT](https://img.shields.io/badge/Model-BioBERT-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

##  Vue d'ensemble
Ce projet est un **système de détection de similarité sémantique** spécialisé pour le domaine médical. Son objectif est de déterminer si deux questions posées par des patients, bien que formulées différemment, ont la même intention médicale (ex: *"Quels sont les symptômes de la grippe ?"* vs *"Comment savoir si j'ai attrapé la grippe ?"*).

Il repose sur une approche de **Transfer Learning** utilisant le modèle **BioBERT** (Bidirectional Encoder Representations from Transformers for Biomedical Text Mining), ré-entraîné (fine-tuned) sur un jeu de données de paires de questions médicales.

###  Fonctionnalités Clés
* **NLP de Pointe (State-of-the-Art) :** Propulsé par le modèle `dmis-lab/biobert-base-cased-v1.2`.
* **Support Multilingue 🌍 :** Accepte les questions en **Français** et en **Anglais**. Un pipeline de traduction (Google Translate API) convertit automatiquement les requêtes vers l'anglais pour l'analyse, rendant l'outil accessible aux francophones.
* **Haute Précision :** Performance atteinte de **~83.7% d'accuracy** et **0.84 de F1-Score** sur le jeu de test.
* **Interface Interactive :** Application Web intuitive développée avec Streamlit.
* **Transparence (Explainability) :** Affiche les scores de confiance (probabilités) et la traduction interne utilisée par le modèle.

##  Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/Ibrahima-Camara20/D-tection-de-Similarit-dans-les-Questions-M-dicales.git](https://github.com/Ibrahima-Camara20/D-tection-de-Similarit-dans-les-Questions-M-dicales.git)
    cd medical-similarity-biobert
    ```

2.  **Installer les dépendances :**
    Assurez-vous d'avoir Python installé, puis lancez :
    ```bash
    pip install -r requirements.txt
    ```

3.  **Télécharger le Modèle :**
    * Récupérez le dossier du modèle entraîné (contenant `model.safetensors`, `config.json`, etc.).
    * Placez ce dossier à la racine du projet et renommez-le impérativement : **`BERT_for_Question_Pair`**.

## Utilisation

Pour lancer l'interface web, exécutez la commande suivante dans votre terminal :

```bash
streamlit run app.py