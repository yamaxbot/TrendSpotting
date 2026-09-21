from django.urls import path

from . import views


urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("trend/<slug:slug>/", views.trend_detail, name="trend_detail"),
]
