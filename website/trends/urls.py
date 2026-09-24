from django.urls import path

from . import views


urlpatterns = [
    path("search/status/", views.search_status, name="search_status"),
    path("", views.dashboard, name="dashboard"),
    path("trend/<slug:slug>/", views.trend_detail, name="trend_detail"),
]
