from django.db import models

# Create your models here.


class Testing(models.Model):
    first_name = models.CharField(max_length=35)
    last_name = models.CharField(max_length=35)
    picture_type = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)


class TestingImages(models.Model):
    file_name = models.CharField(max_length=200)
    detect = models.ForeignKey(
        Testing, on_delete=models.CASCADE)
