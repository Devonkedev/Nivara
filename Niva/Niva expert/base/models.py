# base/models.py
from django.db import models

class ArticleEvaluation(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    summary = models.TextField()
    misinformation_category = models.CharField(
        max_length=50,
        choices=[
            ('major_misinformation', 'Major Misinformation'),
            ('minor_inaccuracies', 'Minor Inaccuracies'),
            ('recommended', 'Recommended Article')
        ]
    )
    rating = models.IntegerField()  # Rating out of 5
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
