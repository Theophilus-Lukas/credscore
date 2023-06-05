import imp
from pyexpat import model
from rest_framework import serializers
from .models import Image


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'user_name', 'created_at',
                  'updated_at', 'image']
