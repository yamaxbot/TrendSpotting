from django.urls import path

from . import views


urlpatterns = [
    path("search/status/", views.search_status, name="search_status"),
    path("", views.dashboard, name="dashboard"),
    path("how-it-works/", views.how_it_works, name="how_it_works"),
    path("about/", views.about_project, name="about_project"),
    path("trend/<slug:slug>/", views.trend_detail, name="trend_detail"),
    path("trend/<slug:slug>/evidence-status/", views.evidence_status, name="evidence_status"),
]
