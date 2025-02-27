"""hackathon URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from api import views
from django.conf.urls.static import static

from hackathon import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^$', views.index, name='index'),
    url(r'^api/search/$', views.PhotoSearchList.as_view(), name='search-photo'),
    url(r'^api/search/(?P<pk>[0-9]+)/$', views.PhotoSearchDetail.as_view(), name='search-photo-detail'),
    url(r'^api/photo/$', views.PhotoList.as_view(), name='photo'),
    url(r'^api/photo/(?P<pk>[0-9]+)/$', views.PhotoDetail.as_view(), name='photo-detail')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
