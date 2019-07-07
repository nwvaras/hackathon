from rest_framework import serializers

from api.models import Photo


class PhotoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Photo
        fields = ('url', 'id', 'image')

    result_image = serializers.SerializerMethodField()


class SearchPhotoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Photo
        fields = ('url', 'id', 'image')

    def get_image(self, obj):
        return obj.image.url.replace(obj.image.name, obj.result_image).replace('photos', 'results')