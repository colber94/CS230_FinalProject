import numpy as np
from PIL import Image

def image_crop(img, size):
  imgwidth, imgheight = img.size
  rows = np.int(imgheight/size)
  cols = np.int(imgwidth/size)
  output_images = []
  for i in range(rows):
      for j in range(cols):
          new_area= (j*size, i*size, (j+1)*size, (i+1)*size)
          new_image = img.crop(new_area)
          output_images.append(new_image)
  return output_images

def load_data(img_files, crop_size=None, resize=None, scale=None):

  imgs = [Image.open(img) for img in img_files]

  if crop_size:
    imgs = [img.resize((5120,5120), Image.ANTIALIAS) for img in imgs]
    cropped_imgs = []
    for img in imgs:
      cropped_imgs.extend(image_crop(img, crop_size))
    imgs = cropped_imgs

  if resize: imgs = [img.resize((resize, resize),Image.ANTIALIAS) for img in imgs]

  imgs = np.asarray([np.asarray(img) for img in imgs])

  if scale: imgs = imgs*scale

  return imgs