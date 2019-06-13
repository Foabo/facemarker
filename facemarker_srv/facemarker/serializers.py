from rest_framework import serializers
from .models import *
class FeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Features
        fields=("id",
                "scholar_id",
                "feature",
                "idx"
                )


class ScholarSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scholar
        fields = (
            "id",
            "name",
            "organization",
            "homepage",
            "pic_url"
        )