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
from api.ffhq_dataset.face_alignment import image_align
from api.ffhq_dataset.landmarks_detector import LandmarksDetector
import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
from api import dnnlib
from api.dnnlib import tflib
from api import config
from api.encoder.generator_model import Generator
from api.encoder.perceptual_model import PerceptualModel
sys.modules['dnnlib'] = dnnlib
URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

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
dnnlib.tflib.init_tf()


def main(filename):
    # parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    # parser.add_argument('src_dir', help='Directory with images for encoding')
    src_dir = "media/aligned_photos/"
    # parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    generated_images_dir = "media/generated_images/"
    # parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')
    dlatent_dir = "media/latent_representations/"
    # for now it's unclear if larger batch leads to better performance/quality
    # parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    batch_size = 1
    # Perceptual model params
    # parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    image_size = 256
    # parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    lr = 1
    # parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)
    iterations = 1000
    # Generator params
    # parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    randomize_noise= False
    # args, other_args = parser.parse_known_args()
    basepath = os.path.dirname(__file__)
    print(os.path.join("..",src_dir))
    ref_images = [os.path.join(basepath,"..",src_dir, x) for x in os.listdir(os.path.join(basepath,"..",src_dir))]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % src_dir)

    os.makedirs(generated_images_dir, exist_ok=True)
    os.makedirs(dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model

    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    print(filename)
    for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

        perceptual_model.set_reference_images(images_batch)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr)
        pbar = tqdm(op, leave=False, total=iterations)
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
        print(' '.join(names), ' loss:', loss)

        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(basepath,"..",generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(basepath,"..",dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()

def align(filename):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    print("align!")
    print(filename)


    from os import path

    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "media/photos"))
    filepath2 = path.abspath(path.join(basepath, "..", "media/aligned_photos/"))
    print(filepath)
    print(filepath2)
    RAW_IMAGES_DIR = filepath
    ALIGNED_IMAGES_DIR = filepath2


    for img_name in os.listdir(RAW_IMAGES_DIR):
        if img_name in filename:
            print("alineando...")
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)

                image_align(raw_img_path, aligned_face_path, face_landmarks)
    main("test")

def index(request):
    return render(request, 'api/index.html')


class PhotoList(APIView):
    permission_classes = (permissions.AllowAny, )

    def post(self, request):
        serializer = PhotoSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            image =serializer.save()
            # print(serializer.save())
            align(image.image.name)

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