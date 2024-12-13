import requests
import pandas as pd
import numpy as np

# Charger le fichier Excel
def load_excel(file_path, sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# Vérification si un embedding contient des NaN ou valeurs infinies
def is_valid_embedding(embedding):
    return embedding is not None and not np.any(np.isnan(embedding)) and not np.any(np.isinf(embedding))

# Utiliser Ollama pour obtenir les embeddings des idées et des critères
def get_embedding(text):
    try:
        response = requests.post(
            "http://localhost:11434/v1/embeddings",
            headers={"Content-Type": "application/json"},
            json={"model": "nomic-embed-text", "input": text}
        )
        
        if response.status_code != 200:
            raise ValueError(f"Erreur dans la requête API Ollama : {response.status_code}, {response.text}")
        
        response_json = response.json()
        
        if 'data' in response_json and len(response_json['data']) > 0:
            embedding = response_json['data'][0]['embedding']
            if is_valid_embedding(embedding):
                return embedding
            else:
                return None
        else:
            raise ValueError("L'API Ollama ne renvoie pas les données attendues.")
    
    except Exception as e:
        return None

# Fonction pour évaluer les idées avec Ollama
def evaluate_ideas_with_ollama(ideas_df, criteria_df):
    # Vérification des colonnes dans le fichier de critères
    if 'Sous-Critères' not in criteria_df.columns or 'Poids Total (sur 7)' not in criteria_df.columns or 'Question Clé' not in criteria_df.columns:
        raise ValueError("Les colonnes attendues ('Sous-Critères', 'Poids Total (sur 7)', 'Question Clé') sont manquantes dans le fichier de critères.")
    
    # Extraire les poids des critères et les questions
    criteria_weights = criteria_df.set_index('Sous-Critères')['Poids Total (sur 7)'].to_dict()
    criteria_questions = criteria_df.set_index('Sous-Critères')['Question Clé'].to_dict()
    
    # Créer les embeddings pour les critères
    criteria_embeddings = {}
    for criterion, question in criteria_questions.items():
        embedding = get_embedding(question)
        if embedding is not None:
            criteria_embeddings[criterion] = embedding
    
    if not criteria_embeddings:
        raise ValueError("Aucun embedding valide n'a pu être généré pour les critères.")
    
    # Créer les embeddings pour les idées
    idea_descriptions = ideas_df['Description de l\'idée (Décrivez votre idée en détail)'].fillna('').tolist()
    idea_embeddings = []
    for description in idea_descriptions:
        if description:
            embedding = get_embedding(description)
            if embedding is not None:
                idea_embeddings.append(embedding)
    
    if not idea_embeddings:
        raise ValueError("Aucun embedding valide n'a pu être généré pour les idées.")
    
    scores = []
    commentaires = []
    
    # Calcul de la similarité pour chaque idée et critère
    for idea_embedding in idea_embeddings:
        if idea_embedding is None:
            continue  # Si l'embedding de l'idée est invalide, passer à l'itération suivante
        total_score = 0
        idea_commentaire = []
        for criterion, weight in criteria_weights.items():
            criterion_embedding = criteria_embeddings.get(criterion)
            if criterion_embedding is None:
                continue  # Si l'embedding du critère est invalide, passer à l'itération suivante
            # Calcul de la similarité cosinus
            similarity = np.dot(idea_embedding, criterion_embedding) / (np.linalg.norm(idea_embedding) * np.linalg.norm(criterion_embedding))
            if np.isnan(similarity) or np.isinf(similarity):
                similarity = 0  # Si la similarité est invalide, la définir à 0
            # Appliquer un score sur 7 en fonction du poids
            contribution = (similarity * weight) / 7 * 20   # Normaliser sur 20
            total_score += contribution
            # Explication détaillée dans le commentaire
            idea_commentaire.append(f"Critère: {criterion}, Similarité: {similarity:.2f}, Poids: {weight}, Contribution: {contribution:.2f}")
        
        scores.append(total_score)
        commentaires.append(" | ".join(idea_commentaire))

    # Ajouter les scores et les commentaires aux idées, puis les trier
    ideas_df['Score'] = scores
    ideas_df['Commentaire'] = commentaires
    ranked_ideas = ideas_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    return ranked_ideas

# Charger les données
ideas_file = "TemplateCONCOURS INNO 2024.xlsx"
criteria_file = "critere1.xlsx"

ideas_df = load_excel(ideas_file)
criteria_df = load_excel(criteria_file)

# Évaluer les idées avec Ollama
try:
    ranked_ideas = evaluate_ideas_with_ollama(ideas_df, criteria_df)
    # Enregistrer le classement des idées dans un fichier Excel
    output_file = "ordre_idées1.xlsx"
    ranked_ideas.to_excel(output_file, index=False)
    
    print(f"Les idées ont été classées et sauvegardées dans le fichier : {output_file}")
except Exception as e:
    print(f"Erreur dans le traitement des idées : {e}")
