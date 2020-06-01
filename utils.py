import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2
import matplotlib.pyplot as pp
import seaborn as sns

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def crop_array(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncrop_array(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def load_data(img_files, crop_size=None, resize=None, scale=None):
    
    imgs = []
    for f in img_files:
        img = cv2.imread(f,cv2.IMREAD_UNCHANGED)
        if resize: img = cv2.resize(img,(resize,resize))
        if img.ndim < 3: img = np.expand_dims(img,2)
        if crop_size: img = crop_array(img,(crop_size,crop_size,img.shape[-1]))
        if scale: img = img*scale
        imgs.extend(img)
    return np.asarray(imgs)

def load_sat_imgs(sat_files, cell_files=None, resize=None, crop_size=None, scale=1./256.):
  X = load_data(sat_files,crop_size=crop_size,resize=resize,scale=scale)

  if cell_files is not None:
    cell_arr = load_data(cell_files,crop_size=crop_size,resize=resize)
    X = np.concatenate((X,cell_arr),3)

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
        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)

    # Split into train and dev sets
    if batch_size: idx = np.int(batch_size*np.round((m*split)/batch_size))
    else: idx = np.int(np.round(m*split))

    X_train = X[:idx]
    Y_train = Y[:idx]
    X_dev = X[idx:]
    Y_dev = Y[idx:]

    return (X[:idx], Y[:idx]), (X[idx:], Y[idx:])

def plot_metrics(model_history,figsize=(10,10)):
    pp.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    pp.subplot(311)
    pp.plot(model_history.history["loss"],label="loss",linewidth=2)
    pp.plot(model_history.history["val_loss"],label="val_loss",linewidth=2)
    pp.title("loss");
    pp.ylabel("categorical cross entropy");
    pp.xlabel("epochs");
    pp.legend();
    
    f1_score = lambda p,r: 2*p*r/(p+r)
    f1_train = f1_score(np.asarray(model_history.history["Precision"]),np.asarray(model_history.history["Recall"]))
    f1_val = f1_score(np.asarray(model_history.history["val_Precision"]),np.asarray(model_history.history["val_Recall"]))
    
    sns.set_palette("Paired")
    pp.subplot(312)
    pp.plot(f1_train,label="f1",linewidth=2)
    pp.plot(f1_val,label="val_f1",linewidth=2)
    pp.legend();
    pp.title("f1 score");

    pp.subplot(313)
    pp.plot(model_history.history["Precision"],label="Precision",linewidth=2)
    pp.plot(model_history.history["val_Precision"],label="val_Precision",linewidth=2)
    pp.plot(model_history.history["Recall"],label="Recall",linewidth=2)
    pp.plot(model_history.history["val_Recall"],label="val_Recall",linewidth=2)
    pp.title("precision & recall");
    pp.xlabel("epochs");
    pp.legend();
    
    pp.tight_layout()

def plot_cropped_examples(model, train_set, dev_set, resize, crop, nclasses):
    (X_train, Y_train) = train_set
    (X_dev, Y_dev) = dev_set
    
    
    crops = int((resize/crop)**2)
    select_train = np.random.choice(range(len(X_train)),crops)
    select_dev = np.random.choice(range(len(X_dev)),crops)
    
    yhat_dev = model.predict(X_dev[select_dev])
    yhat_train = model.predict(X_train[select_train])
    
    pp.figure(figsize=(10,10))

    pp.subplot(221)
    pp.title("ground truth")
    masks = uncrop_array(Y_train[select_train,:,:,:],(resize,resize,nclasses)).argmax(-1)
    pp.imshow(masks,cmap="viridis")
    pp.ylabel("train");
    pp.xticks([])
    pp.yticks([])


    pp.subplot(222)
    pp.title("prediction")
    yhat_train_uncropped = uncrop_array(yhat_train,(resize,resize,nclasses)).argmax(-1)
    pp.imshow(yhat_train_uncropped,cmap="viridis")
    pp.xticks([])
    pp.yticks([])

    pp.subplot(223)
    masks = uncrop_array(Y_dev[select_dev,:,:,:],(resize,resize,nclasses)).argmax(-1)
    pp.imshow(masks,cmap="viridis")
    pp.ylabel("validation");
    pp.xticks([])
    pp.yticks([])


    pp.subplot(224)
    yhat_dev_uncropped = uncrop_array(yhat_dev,(resize,resize,nclasses)).argmax(-1)
    pp.imshow(yhat_dev_uncropped,cmap="viridis")
    pp.xticks([])
    pp.yticks([])

    pp.tight_layout()
    
def plot_uncropped_examples(model, sat_files, cell_files, lte_files, resize, crop, nclasses, figsize=(15,15)):
    crops = int((resize/crop)**2)
    
    nexamples = len(sat_files)
    sats_cropped = load_sat_imgs(sat_files,cell_files,resize=resize, crop_size=crop)
    sats = [uncrop_array(sats_cropped[crops*s:crops*s+crops,:,:,:3],(resize,resize,nexamples)) for s in range(nexamples)]
    lte_cropped = load_masks(lte_files,resize=resize, crop_size=crop, nclasses=nclasses)
    lte = [uncrop_array(lte_cropped[crops*s:crops*s+crops,:,:,:],(resize,resize,nclasses)).argmax(-1) for s in range(nexamples)]
    predict_cropped = model.predict(sats_cropped)
    predict = [uncrop_array(predict_cropped[crops*s:crops*s+crops,:,:,:],(resize,resize,nclasses)).argmax(-1) for s in range(nexamples)]
    
    pp.figure(figsize=figsize)
    for i,x,y,h in zip(range(nexamples),sats,lte, predict):
        pp.subplot(nexamples,nexamples,1+nexamples*i);
        if i ==0: pp.title("sentinel-2 image")
        pp.imshow(x,cmap="terrain");
        pp.xticks([])
        pp.yticks([])
        pp.ylabel(sat_files[i].split("/")[-1]);
        
        pp.subplot(nexamples,nexamples,2+nexamples*i);
        if i ==0: pp.title("lte coverage")
        pp.imshow(y,cmap="viridis");
        pp.xticks([])
        pp.yticks([])

        pp.subplot(nexamples,nexamples,nexamples+nexamples*i);
        if i ==0: pp.title("predicted coverage")
        pp.imshow(h,cmap="viridis");
        pp.xticks([])
        pp.yticks([])
    
    pp.tight_layout()
    

