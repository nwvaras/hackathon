import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
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

    ref_images = [os.path.join("..",src_dir, x) for x in os.listdir(os.path.join("..",src_dir))]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % src_dir)

    os.makedirs(generated_images_dir, exist_ok=True)
    os.makedirs(dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
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
            img.save(os.path.join("..",generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join("..",dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()


if __name__ == "__main__":
    main()
