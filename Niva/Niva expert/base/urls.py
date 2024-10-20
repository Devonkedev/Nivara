from django.urls import path, include
from .views import authView, article_finder, test_feed, evaluate_article
from django.contrib import admin
from django.urls import path, include
from django.urls import path


urlpatterns = [
    path("", article_finder, name="article_finder"),
    path('evaluate_article/<str:title>/<path:url>/', evaluate_article, name='evaluate_article'),
    path('test/', test_feed, name='test_feed'),  # Temporary testing URL
    path("signup/", authView, name="authView"),
    path("accounts/", include("django.contrib.auth.urls")),
]
