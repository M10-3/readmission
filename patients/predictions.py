import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from .models import Patient

# Fonction pour augmenter les données existantes
def augment_data(df):
    augmented_data = []
    for _, row in df.iterrows():
        for _ in range(5):  # Crée 5 variations pour chaque patient
            augmented_row = row.copy()
            augmented_row['poids'] += random.uniform(-2, 2)  # Variation de ±2 kg
            augmented_row['taille'] += random.uniform(-2, 2)  # Variation de ±2 cm
            augmented_row['imc'] = round(augmented_row['poids'] / ((augmented_row['taille'] / 100) ** 2), 2)
            augmented_data.append(augmented_row)
    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df

# Préparation des données
def prepare_data():
    # Récupérer les données des patients
    patients = Patient.objects.all()
    data = []
    for patient in patients:
        data.append({
            'num_id': patient.num_id,
            'prenom': patient.prenom,
            'nom': patient.nom,
            'email': patient.email,
            'dob': patient.dob,
            'genre': patient.genre,
            'numero': patient.numero,
            'adresse': patient.adresse,
            'antecedents_medicaux': patient.antecedents_medicaux if patient.antecedents_medicaux else '',  # Assurer une valeur vide si null
            'age': 2024 - patient.dob.year,  # Calculer l'âge
            'poids': patient.poids if patient.poids else 0,  # Assurer une valeur par défaut
            'taille': patient.taille if patient.taille else 0,  # Assurer une valeur par défaut
        })

    df = pd.DataFrame(data)

    # Convertir les variables catégorielles en numériques
    df['genre'] = df['genre'].map({'M': 0, 'F': 1, 'O': 2})

    # Calcul de l'IMC
    df['taille_m'] = df['taille'] / 100  # Convertir la taille en mètres
    df['imc'] = df.apply(lambda x: round(x['poids'] / (x['taille_m'] ** 2), 2) if x['taille_m'] > 0 else 0, axis=1)

    # Exemple de création d'une variable cible (label) : prédire la réadmission
    df['antecedents_medicaux'] = df['antecedents_medicaux'].fillna('')
    df['readmission'] = df['antecedents_medicaux'].apply(lambda x: 1 if 'diabète' in x.lower() else 0)

    # Augmenter les données
    augmented_df = augment_data(df)

    return augmented_df

# Fonction pour entraîner un modèle avec validation croisée
def train_model_with_cross_validation():
    # Préparer les données
    df = prepare_data()
    X = df[['age', 'genre', 'poids', 'taille', 'imc']]  # Caractéristiques
    y = df['readmission']  # Cible

    # Diviser les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle
    model = LogisticRegression()

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédictions probabilistes sur les données de test
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Validation croisée (k=5 folds) pour la précision moyenne
    accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(accuracy_scores)

    # Générer des prédictions via validation croisée
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Calcul des métriques avancées
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Retourner les valeurs nécessaires
    return model, mean_accuracy, f1, roc_auc, X_test, y_test, y_pred_proba

# Exemple d'appel de la fonction pour entraîner le modèle
if __name__ == "__main__":
    model, mean_accuracy, f1, roc_auc = train_model_with_cross_validation()
    print(f"Mean Cross-Validation Accuracy: {mean_accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC Score: {roc_auc}")
