from django.http import Http404
from django.shortcuts import render

# Create your views here.
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from api.models import Photo
from api.serializers import PhotoSerializer
import subprocess

import os
import sys
import bz2
from keras.utils import get_file
from .ffhq_dataset.face_alignment import image_align
from .ffhq_dataset.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path
landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
landmarks_detector = LandmarksDetector(landmarks_model_path)

def align(filename):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    print("align!")
    print(filename)


    from os import path

    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "media/photos/"))
    filepath2 = path.abspath(path.join(basepath, "..", "media/aligned_photos/"))
    print(filepath)
    print(filepath2)
    RAW_IMAGES_DIR = filepath
    ALIGNED_IMAGES_DIR = filepath2


    for img_name in os.listdir(RAW_IMAGES_DIR):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)

            image_align(raw_img_path, aligned_face_path, face_landmarks)

def index(request):
    return render(request, 'api/index.html')


class PhotoList(APIView):
    permission_classes = (permissions.AllowAny, )

    def post(self, request):
        serializer = PhotoSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            align(serializer.data.image.name)

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PhotoDetail(APIView):

    permission_classes = (permissions.AllowAny,)

    def get_object(self, pk):
        try:
            return Photo.objects.get(pk=pk)
        except Photo.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        photo = self.get_object(pk)
        serializer = PhotoSerializer(photo)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        photo = self.get_object(pk)
        serializer = PhotoSerializer(photo, data=request.DATA)
        if serializer.is_valid():
            serializer.save()
            # print(subprocess.run(["python align_images.py ../api/media/photos/ ../api/media/aligned_photos/"]))
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        photo = self.get_object(pk)
        photo.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    def pre_save(self, obj):
        obj.owner = self.request.user