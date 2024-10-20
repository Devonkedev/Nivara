# base/forms.py
from django import forms
from .models import ArticleEvaluation

class ArticleEvaluationForm(forms.ModelForm):
    class Meta:
        model = ArticleEvaluation
        fields = ['summary', 'misinformation_category', 'rating']
        widgets = {
            'summary': forms.Textarea(attrs={'rows': 4, 'cols': 40}),
            'rating': forms.NumberInput(attrs={'min': 1, 'max': 5}),
        }
