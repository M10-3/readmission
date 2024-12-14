from django.db import models
import random
import string
from datetime import date

class Patient(models.Model):
    GENRE_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
     
    prenom = models.CharField(max_length=50)
    nom = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    num_id = models.CharField(max_length=20, unique=True, editable=False)
    dob = models.DateField()
    numero = models.CharField(max_length=15)
    adresse = models.CharField(max_length=255)
    antecedents_medicaux = models.TextField(blank=True, null=True)
    genre = models.CharField(max_length=1, choices=GENRE_CHOICES, default='M')
    photo_profil = models.ImageField(upload_to='profiles/', blank=True, null=True)
    poids = models.FloatField(null=True, blank=True)
    taille = models.FloatField(null=True, blank=True)

    @property
    def age(self):
        today = date.today()
        return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))

    @property
    def imc(self):
        if self.poids and self.taille:
            taille_metre = self.taille / 100
            return round(self.poids / (taille_metre ** 2), 2)
        return None

    def save(self, *args, **kwargs):
        if not self.num_id:
            initials = f"{self.prenom[0].upper()}{self.nom[0].upper()}"
            birth_year = self.dob.year
            random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            self.num_id = f"{initials}{birth_year}_{random_part}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.nom} {self.prenom}"
