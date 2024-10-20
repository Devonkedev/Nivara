from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
import feedparser
from django.views.generic import ListView, DetailView
from django.http import HttpResponse
from .forms import ArticleEvaluationForm
from .models import ArticleEvaluation

@login_required
def article_finder(request):
    # URL of the RSS feed
    feed_url = "https://feeds.bbci.co.uk/news/rss.xml"
    
    # Fetch and parse the feed
    feed = feedparser.parse(feed_url)
    
    # Extract entries from the feed
    articles = feed.entries


   # If a search term is provided
    query = request.GET.get('q')
    if query:
        articles = [article for article in articles if query.lower() in article.title.lower()]

    return render(request, 'articles/article_finder.html', {'articles': articles})


def test_feed(request):
    feed_url = "https://rss.cnn.com/rss/edition.rss"
    feed = feedparser.parse(feed_url)
    
    articles = feed.entries
    
    # Return the raw article titles as plain text for testing
    output = ""
    for article in articles:
        output += f"{article.title}\n"
    
    return HttpResponse(output)



def evaluate_article(request, title, url):
    if request.method == 'POST':
        form = ArticleEvaluationForm(request.POST)
        if form.is_valid():
            evaluation = form.save(commit=False)
            evaluation.title = title  # Save the article title
            evaluation.url = url  # Save the article URL
            evaluation.save()
            return redirect('article_finder')  # Redirect back to the main article page
    else:
        form = ArticleEvaluationForm()

    return render(request, 'base/evaluate_article.html', {
        'form': form,
        'title': title,
        'url': url
    })
"""@login_required
def home(request):
 return render(request, "home.html", {})"""


def authView(request):
 if request.method == "POST":
  form = UserCreationForm(request.POST or None)
  if form.is_valid():
   form.save()
   return redirect("base:login")
 else:
  form = UserCreationForm()
 return render(request, "registration/signup.html", {"form": form})


