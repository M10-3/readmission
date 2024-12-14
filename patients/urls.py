from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.patient_list, name='patient_list'),
    path('add/', views.add_patient, name='add_patient'),
    path('results/', views.model_results_view, name='model_results_view'),
    path('download-report/', views.download_report, name='download_report'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
