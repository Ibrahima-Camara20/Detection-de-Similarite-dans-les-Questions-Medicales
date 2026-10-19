import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Détection de Similarité Médicale",
    layout="centered"
)

# --- CHARGEMENT DU MODÈLE (Depuis Hugging Face) ---
@st.cache_resource
def load_model():
   
    model_path = "Ibrahima20/bert-question-pairs-similarity"
    
    try:
        # Cela va télécharger le modèle automatiquement (environ 400 Mo)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Erreur critique lors du téléchargement du modèle : {e}")
        return None, None

# --- TRADUCTION LOCALE FR -> EN (sans internet, sans limite) ---
@st.cache_resource
def load_translator():
    name = "Helsinki-NLP/opus-mt-fr-en"
    return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name)

@st.cache_data(show_spinner=False)
def to_english(text: str) -> str:
    text = text.strip()
    try:
        if detect(text) == "en":
            return text  # déjà en anglais : on ne traduit pas
    except Exception:
        pass  # langue non détectée : on tente la traduction
    tok, mt = load_translator()
    batch = tok([text], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = mt.generate(**batch, max_new_tokens=128)
    return tok.decode(out[0], skip_special_tokens=True)

# On charge le modèle une seule fois au démarrage
with st.spinner("Chargement du cerveau de l'IA en cours..."):
    tokenizer, model = load_model()

# --- INTERFACE UTILISATEUR ---
st.title(" Assistant de Similarité Médicale")
st.markdown("---")
st.info("Vous pouvez poser vos questions en **Français** ou en **Anglais**.")

# Zone de saisie
col1, col2 = st.columns(2)
with col1:
    question1 = st.text_area("Question 1", placeholder="Ex: J'ai une douleur dans la poitrine...", height=100)
with col2:
    question2 = st.text_area("Question 2", placeholder="Ex: Comment savoir si je fais une crise cardiaque ?", height=100)

# Bouton d'action
if st.button("🔍 Analyser la similarité", use_container_width=True):
    if not question1 or not question2:
        st.warning(" Veuillez remplir les deux questions.")
    else:
        if tokenizer and model:
            
            # --- ÉTAPE 1 : TRADUCTION ---
            with st.spinner("Traduction et analyse en cours..."):
                try:
                    # Traduction locale des deux textes
                    q1_en = to_english(question1)
                    q2_en = to_english(question2)
                    
                    
                    with st.expander("Voir la traduction utilisée par le modèle (Interne)"):
                        st.write(f"**Q1 (EN):** {q1_en}")
                        st.write(f"**Q2 (EN):** {q2_en}")

                    # --- ÉTAPE 2 : TOKENIZATION ---
                    inputs = tokenizer(
                        q1_en,     
                        q2_en,      
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=128
                    )
                    
                    # --- ÉTAPE 3 : PRÉDICTION ---
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probabilities = F.softmax(logits, dim=1)
                    
                    # Récupération des résultats
                    # Rappel : 0 = Différent, 1 = Similaire
                    score_similaire = probabilities[0][1].item()
                    score_different = probabilities[0][0].item()
                    predicted_class = torch.argmax(probabilities, dim=1).item()

                    # --- ÉTAPE 4 : AFFICHAGE ---
                    st.markdown("### Résultat de l'analyse :")
                    
                    if predicted_class == 1:
                        st.success(f"✅ **SIMILAIRES** (Confiance : {score_similaire:.1%})")
                        st.progress(score_similaire)
                        st.info("Le modèle estime que ces deux questions demandent la même information médicale.")
                    else:
                        st.error(f"❌ **DIFFÉRENTES** (Confiance : {score_different:.1%})")
                        st.progress(score_different)
                        st.write("Le modèle estime que ces questions traitent de sujets différents.")
                
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de l'analyse : {e}")