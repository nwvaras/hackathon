from django.db import models

# Create your models here.


class Photo(models.Model):
    image = models.ImageField(upload_to='photos', max_length=254)
    result_image = models.CharField( max_length=254, null=True)
    latent_filename = models.CharField(max_length=254)