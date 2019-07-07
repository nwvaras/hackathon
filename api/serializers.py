from rest_framework import serializers

from api.models import Photo


class PhotoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = ('id', 'image')


class SearchPhotoSerializer(serializers.ModelSerializer):
    image = serializers.SerializerMethodField()

    class Meta:
        model = Photo
        fields = ('id', 'image')

    def get_image(self, obj):
        return obj.image.url.replace(obj.image.name, obj.result_image).replace('photos', 'results')