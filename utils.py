import numpy as np
from tensorflow.keras.utils import to_categorical
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

def load_sat_imgs(sat_files, cell_files=None, crop_size=None, resize=None):
  X = load_data(sat_files,crop_size=crop_size,resize=resize,scale=1./256.)

  if cell_files:
    cell_arr = load_data(cell_files,crop_size=crop_size,resize=resize)
    X = np.concatenate((X,np.expand_dims(cell_arr,3)),3)

  return X

def load_masks(mask_files,nclasses,crop_size=None, resize=None, onehot=True):
  Y = load_data(mask_files,crop_size=crop_size,resize=resize)
  Y[Y > nclasses-1] = nclasses - 1 # in case something weird happens with antialiasing
  if onehot: Y = to_categorical(Y,nclasses)

  return Y

def split_train_dev(X,Y,split=0.8,batch_size=None,shuffle=True,seed=None):

    m = len(X)

    # Shuffle arrays along examplar axis
    if shuffle:
        if seed: np.random.seed(seed)
        shuffle = np.random.permutation(m)
        X = X[shuffle]
        Y = Y[shuffle]

    # Split into train and dev sets
    if batch_size: idx = np.int(batch_size*np.round((m*split)/batch_size))
    else: idx = np.int(np.round(m*split))

    X_train = X[:idx]
    Y_train = Y[:idx]
    X_dev = X[idx:]
    Y_dev = Y[idx:]

    return (X_train, Y_train), (X_dev, Y_dev)