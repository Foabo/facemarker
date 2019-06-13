from django.db import models
import requests
import uuid
# Create your models here.
class Features(models.Model):
    id = models.CharField(default=uuid.uuid4(),primary_key=True,db_index=True,max_length=36)
    scholar_id = models.CharField(null=True,max_length=36)
    feature = models.BinaryField()
    idx = models.CharField(max_length=64)


class Scholar(models.Model):
    id = models.CharField(max_length=36,primary_key=True,db_index=True,default=uuid.uuid4())
    name = models.CharField(max_length=255,null=True)
    organization = models.CharField(max_length=255,null=True)
    homepage = models.CharField(max_length=1000,null=True)
    pic_url = models.CharField(max_length=1000,null=True)