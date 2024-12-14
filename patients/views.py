from django.shortcuts import render, redirect
from .models import Patient
from .forms import PatientForm
from django.shortcuts import render
from .models import Patient
from .predictions import train_model_with_cross_validation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from io import BytesIO
import base64
import csv
from django.http import HttpResponse

def patient_list(request):
    patients = Patient.objects.all()
    return render(request, 'patients/patient_list.html', {'patients': patients})

def add_patient(request):
    if request.method == 'POST':
        form = PatientForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('patient_list')
    else:
        form = PatientForm()
    return render(request, 'patients/add_patient.html', {'form': form})


#ia 


def model_results_view(request):
    if request.method == 'POST':
        # Entraîner le modèle et obtenir les métriques et prédictions
        model, mean_accuracy, f1, roc_auc, X_test, y_test, y_pred_proba = train_model_with_cross_validation()

        # Chargement des données des patients
        patients = Patient.objects.all()
        data = [{'age': 2024 - p.dob.year, 'imc': p.imc} for p in patients if p.taille and p.poids]
        df = pd.DataFrame(data)

        # Récupérer un patient par défaut (ou connecté via un formulaire, par exemple)
        patient_id = request.POST.get('patient_id')
        connected_patient = Patient.objects.filter(id=patient_id).first()

        patient_message = ""
        patient_recommendation = ""

        if connected_patient and connected_patient.taille and connected_patient.poids:
            # Calculer les caractéristiques du patient sélectionné
            patient_data = {
                'age': connected_patient.age,
                'genre': {'M': 0, 'F': 1, 'O': 2}.get(connected_patient.genre, 2),
                'poids': connected_patient.poids,
                'taille': connected_patient.taille,
                'imc': connected_patient.imc,
            }
            # Convertir en DataFrame pour prédiction
            patient_df = pd.DataFrame([patient_data])
            prob_readmission = model.predict_proba(patient_df)[0][1]  # Probabilité de réadmission
            patient_message = f"Votre probabilité de réadmission est de {prob_readmission:.2%}."

            # Ajouter des recommandations
            if prob_readmission <= 0.30:
                patient_recommendation = "Votre risque de réadmission est faible. Continuez à suivre vos soins habituels."
            elif prob_readmission <= 0.60:
                patient_recommendation = "Votre risque de réadmission est modéré. Pensez à planifier une visite de contrôle."
            else:
                patient_recommendation = "Votre risque de réadmission est élevé. Veuillez consulter un médecin dès que possible."
        else:
            patient_message = "Impossible de calculer votre risque de réadmission : données insuffisantes."

        # Génération de l'histogramme des âges
        plt.figure(figsize=(6, 4))
        plt.hist(df['age'], bins=10, color='skyblue', edgecolor='black')
        plt.title('Répartition des âges')
        plt.xlabel('Âge')
        plt.ylabel('Nombre de patients')
        buffer_age = BytesIO()
        plt.savefig(buffer_age, format='png')
        buffer_age.seek(0)
        graphic_age = base64.b64encode(buffer_age.getvalue()).decode('utf-8')
        buffer_age.close()

        # Génération de la courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc_value = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.legend(loc='lower right')
        buffer_roc = BytesIO()
        plt.savefig(buffer_roc, format='png')
        buffer_roc.seek(0)
        graphic_roc = base64.b64encode(buffer_roc.getvalue()).decode('utf-8')
        buffer_roc.close()

        # Ajout des graphiques et des métriques au contexte
        context = {
            'mean_accuracy': round(mean_accuracy * 100, 2),
            'f1_score': round(f1, 2),
            'roc_auc': round(roc_auc, 2),
            'message': "Le modèle a été entraîné avec succès !",
            'graphic_age': graphic_age,
            'graphic_roc': graphic_roc,
            'patient_message': patient_message,  # Message personnalisé
            'patient_recommendation': patient_recommendation,  # Recommandation personnalisée
            'patients': patients,  # Pour afficher tous les patients dans la vue
        }
    else:
        # Afficher une page de démarrage
        context = {
            'message': "Cliquez sur le bouton pour entraîner le modèle.",
            'patients': Patient.objects.all(),  # Inclure les patients pour sélection
        }

    return render(request, 'patients/model_results.html', context)


def download_report(request):
    # Préparer une réponse HTTP pour un fichier CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="report.csv"'

    writer = csv.writer(response)

    # Ajouter un en-tête au fichier CSV
    writer.writerow(['Numero ID', 'Nom', 'Prenom', 'Age', 'Genre', 'IMC', 'Performance'])

    # Récupérer les patients pour inclure dans le rapport
    patients = Patient.objects.all()

    # Ajoute les performances du modèle si pertinent
    try:
        model, mean_accuracy, f1, roc_auc = train_model_with_cross_validation()
        writer.writerow([])
        writer.writerow(['Performance du modele'])
        writer.writerow(['Precision moyenne (%)', round(mean_accuracy * 100, 2)])
        writer.writerow(['F1 Score', round(f1, 2)])
        writer.writerow(['ROC-AUC Score', round(roc_auc, 2)])
        writer.writerow([])
    except Exception as e:
        writer.writerow([])
        writer.writerow(['Erreur lors de lentraînement du modele'])
        writer.writerow([str(e)])
        writer.writerow([])

    # Ajouter les détails des patients
    for patient in patients:
        genre = dict(Patient.GENRE_CHOICES).get(patient.genre, "N/A")
        imc = patient.imc if patient.imc else "N/A"
        writer.writerow([
            patient.num_id,
            patient.user.nom,
            patient.user.prenom,
            patient.age,
            genre,
            imc,
            'Donnees incluses'
        ])

    return response