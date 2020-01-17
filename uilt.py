import numpy as np
import os
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from bunch import Bunch
import json
import albumentations as albu
from keras.callbacks import Callback
import keras.backend as K
import albumentations.augmentations.functional as F
import cv2
from tqdm import tqdm_notebook
import gc
import random
import pickle
from tqdm import tqdm
from keras.preprocessing.image import load_img

# repeat images in the dataset
gamma = 1.2
inverse_gamma = 1.0 / gamma
look_up_table = np.array([((i / 255.0) ** inverse_gamma) * 255.0 for i in np.arange(0, 256, 1)]).astype("uint8")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

DUPLICATE=np.array([
    'train_images/6eb8690cd.jpg', 'train_images/a67df9196.jpg',
    'train_images/24e125a16.jpg', 'train_images/4a80680e5.jpg',
    'train_images/a335fc5cc.jpg', 'train_images/fb352c185.jpg',
    'train_images/c35fa49e2.jpg', 'train_images/e4da37c1e.jpg',
    'train_images/877d319fd.jpg', 'train_images/e6042b9a7.jpg',
    'train_images/618f0ff16.jpg', 'train_images/ace59105f.jpg',
    'train_images/ae35b6067.jpg', 'train_images/fdb5ae9d4.jpg',
    'train_images/3de8f5d88.jpg', 'train_images/a5aa4829b.jpg',
    'train_images/3bd0fd84d.jpg', 'train_images/b719010ac.jpg',
    'train_images/24fce7ae0.jpg', 'train_images/edf12f5f1.jpg',
    'train_images/49e374bd3.jpg', 'train_images/6099f39dc.jpg',
    'train_images/9b2ed195e.jpg', 'train_images/c30ecf35c.jpg',
    'train_images/3a7f1857b.jpg', 'train_images/c37633c03.jpg',
    'train_images/8c2a5c8f7.jpg', 'train_images/abedd15e2.jpg',
    'train_images/b46dafae2.jpg', 'train_images/ce5f0cec3.jpg',
    'train_images/5b1c96f09.jpg', 'train_images/e054a983d.jpg',
    'train_images/3088a6a0d.jpg', 'train_images/7f3181e44.jpg',
    'train_images/dc0c6c0de.jpg', 'train_images/e4d9efbaa.jpg',
    'train_images/488c35cf9.jpg', 'train_images/845935465.jpg',
    'train_images/3b168b16e.jpg', 'train_images/c6af2acac.jpg',
    'train_images/05bc27672.jpg', 'train_images/dfefd11c4.jpg',
    'train_images/048d14d3f.jpg', 'train_images/7c8a469a4.jpg',
    'train_images/a1a0111dd.jpg', 'train_images/b30a3e3b6.jpg',
    'train_images/d8be02bfa.jpg', 'train_images/e45010a6a.jpg',
    'train_images/caf49d870.jpg', 'train_images/ef5c1b08e.jpg',
    'train_images/63c219c6f.jpg', 'train_images/b1096a78f.jpg',
    'train_images/76096b17b.jpg', 'train_images/d490180a3.jpg',
    'train_images/bd0e26062.jpg', 'train_images/e7d7c87e2.jpg',
    'train_images/600a81590.jpg', 'train_images/eb5aec756.jpg',
    'train_images/ad5a2ea44.jpg', 'train_images/e9fa75516.jpg',
    'train_images/6afa917f2.jpg', 'train_images/9fb53a74b.jpg',
    'train_images/59931eb56.jpg', 'train_images/e7ced5b76.jpg',
    'train_images/0bfe252d0.jpg', 'train_images/b4d0843ed.jpg',
    'train_images/67fc6eeb8.jpg', 'train_images/c04aa9618.jpg',
    'train_images/741a5c461.jpg', 'train_images/dae3c563a.jpg',
    'train_images/78416c3d0.jpg', 'train_images/e34f68168.jpg',
    'train_images/0d258e4ae.jpg', 'train_images/72322fc23.jpg',
    'train_images/0aafd7471.jpg', 'train_images/461f83c57.jpg',
    'train_images/38a1d7aab.jpg', 'train_images/8866a93f6.jpg',
    'train_images/7c5b834b7.jpg', 'train_images/dea514023.jpg',
    'train_images/32854e5bf.jpg', 'train_images/530227cd2.jpg',
    'train_images/1b7d7eec6.jpg', 'train_images/f801dd10b.jpg',
    'train_images/46ace1c15.jpg', 'train_images/876e74fd6.jpg',
    'train_images/578b43574.jpg', 'train_images/9c5884cdd.jpg',
]).reshape(-1,2).tolist()

def plot_history(history):
    plt.figure(figsize = (8,6))
    plt.plot(history.history['loss'], '-', label = 'train_loss', color = 'g')
    plt.plot(history.history['val_loss'], '--', label = 'valid_loss', color ='r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss on unet')
    plt.legend()
    plt.show()

    plt.figure(figsize = (8,6))
    plt.plot(history.history['Dice_coef'], '-', label = 'train_Dice_coef', color = 'g')
    plt.plot(history.history['val_Dice_coef'], '--', label = 'valid_Dice_coef', color ='r')
    plt.xlabel('epoch')
    plt.ylabel('Dice_Coef')
    plt.title('Dice_Coef on unet')
    plt.legend()
    plt.show()

def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = Bunch(config_dict)
    return config

def np_resize(img, input_shape):
    height, width = input_shape
    return cv2.resize(img, (width, height))

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask= np.zeros( width*height ).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T

def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))

    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask

    return masks

def run_length_encode(mask):
    #possible bug for here
    m = mask.T.flatten()
    if m.sum()==0:
        rle=''
    else:
        m   = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
    return rle

def read_data(spilt):
    path =r'J:\kaggle competition\stealdefects\data'
    read_path = os.path.join(path,spilt)
    cvs_path =os.path.join(path,'train.csv')
    id = list(np.load(read_path))
    df = pd.read_csv(cvs_path)
    df.fillna('', inplace=True)
    df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
    df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
    df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in id for c in [1,2,3,4]])
    image_array = np.zeros([len(id),256,1600,3],dtype=np.float32)
    mask_array = np.zeros([len(id),256,1600,4],dtype=np.int)
    label_array = []
    for i in range(len(id)):
        folder, image_id = id[i].split('/')
        rle = [
            df.loc[df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = np.array(cv2.imread(path+'/%s/%s'%(folder,image_id),cv2.IMREAD_COLOR))
        mask = np.array(build_masks(rle, (256,1600)))
        label = (mask.reshape(4,-1).sum(1)>8).astype(np.int32)
        label_array.append(label)
        image_array[i,:,:,:] = image
        mask_array[i,:,:,:] = mask
    return image_array, mask_array,label_array



def print_data_info(df,id):
    num1 = (df['Class']==1).sum()
    num2 = (df['Class']==2).sum()
    num3 = (df['Class']==3).sum()
    num4 = (df['Class']==4).sum()
    pos1 = ((df['Class']==1) & (df['Label']==1)).sum()
    pos2 = ((df['Class']==2) & (df['Label']==1)).sum()
    pos3 = ((df['Class']==3) & (df['Label']==1)).sum()
    pos4 = ((df['Class']==4) & (df['Label']==1)).sum()
    length=len(id)
    num = 4*length
    pos = (df['Label']==1).sum()
    neg = num - pos

def rle_decoding(rle, mask_shape=(256, 1600)):
    strs = rle.split(' ')
    starts = np.asarray(strs[0::2], dtype=int) - 1
    lengths = np.asarray(strs[1::2], dtype=int)
    ends = starts + lengths

    mask = np.zeros(mask_shape[0] * mask_shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        mask[s:e] = 1
    return mask.reshape(mask_shape, order='F')

def merge_masks(image_id, df, mask_shape=(256, 1600), reshape=None):
    rles = df[df['ImageId'] == image_id].EncodedPixels.iloc[:]
    depth = rles.shape[0]
    if reshape:
        masks = np.zeros((*reshape, depth), dtype=np.uint8)
    else:
        masks = np.zeros((mask_shape[0], mask_shape[1], depth), dtype=np.uint8)

    for idx in range(depth):
        if isinstance(rles.iloc[idx], str):
            if reshape:
                cur_mask = rle_decoding(rles.iloc[idx], mask_shape)
                cur_mask = cv2.resize(cur_mask, (reshape[1], reshape[0]))
                masks[:, :, idx] += cur_mask
            else:
                masks[:, :, idx] += rle_decoding(rles.iloc[idx], mask_shape)
    mask_merge = np.sum(masks, axis=2)
    # background = np.array(mask_merge>0, dtype = np.uint8)
    # masks[:,:,4]= 1-background
    return masks

def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort = pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    #df = df.reset_index()
    df = df.drop('sort', axis=1)
    return df


def read(spilt):
    path =os.path.join(os.getcwd(),'data')
    new_id=[]
    read_path = os.path.join(path,spilt)
    cvs_path =os.path.join(path,'train.csv')
    id = list(np.load(read_path))
    for i in range(len(id)):
        folder, image_id = id[i].split('/')
        new_id.append(image_id)
    return new_id


def predict_mask(img,ground_truth):
    # run a grid search for threshold and min mask area
    search_pixel_threshold = [0.25,0.40,0.50,0.65,0.70,0.80,0.85,0.90,0.95]
    search_size_threshold  = np.stack([
                    np.linspace(0,3200,17),
                    np.linspace(0,3200,17),
                    np.linspace(0,6400,17),
                    np.linspace(0,6400,17),
        ])
    class_params = {}
    dice = np.zeros([4,9,17])
    for class_id in range(4):
        for th in range(len(search_pixel_threshold)):
            for minisize in range(len(search_size_threshold[class_id])):
                d=0
                for i in range(img.shape[0]):
                    pred =img[i,:,:,class_id]
                    pred =np.array(pred>search_pixel_threshold[th],np.uint8)
                    pred = cv2.resize(pred, (256, 1600))
                    pred = single_mask_reduce(pred,search_size_threshold[class_id,minisize])

                    if pred.sum()==0 and np.sum(ground_truth[i,:,:,class_id])==0:
                        d+=1
                    else:
                        d+=Dice(pred,ground_truth[i,:,:,class_id])
                dice[class_id,th,minisize]=d/img.shape[0]/1000
    return dice

def Dice(y_true, y_pred, smooth = 1):

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def single_mask_reduce(mask, minSize):
    label_num, label_mask = cv2.connectedComponents(mask.astype(np.uint8))
    reduced_mask = np.zeros(mask.shape, np.float32)
    for label in range(1, label_num):
        single_label_mask = (label_mask == label)
        if single_label_mask.sum() > minSize:
            reduced_mask[single_label_mask] = 1

    return reduced_mask


def contrast_enhancement(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    return img


def gamma_correction(img):
    return cv2.LUT(img.astype('uint8'), look_up_table)

def crop_img(img):
    img_array = np.empty([7, 256, 400, img.shape[-1]], dtype=np.float32)
    for i in range(7):
        img_array[i, :, :, :] = img[:, i * 200:i * 200 + 400, :]
    return img_array

def random_clip(image, mask, shape=(256, 400)):
    a = random.randint(0, 1000)
    crop_img = image[:, a:a + shape[1], :]
    crop_mask = mask[:, a:a + shape[1], :]
    return crop_img, crop_mask

def input_gen(filenames, segs, aug, load_dir, batch_size = 4, reshape = (256,1600)):
    batch_rgb = []
    batch_mask = []

    while True:
        fns = random.sample(filenames, batch_size)
        seed = np.random.choice(range(999))
        for fn in fns:
            cur_img = np.asarray(load_target_image(os.path.join(load_dir,fn),target_size=reshape))
            cur_img = gamma_correction(cur_img)
            cur_img = contrast_enhancement(cur_img)
            masks = merge_masks(fn, segs, reshape = reshape)
            processed = aug(image = cur_img, mask = masks)

            batch_rgb.append(processed['image']/255.0)
            batch_mask.append(processed['mask'])

        batch_rgb, batch_mask = np.stack(batch_rgb), np.stack(batch_mask)

        yield batch_rgb, batch_mask
        batch_rgb = []
        batch_mask = []

def load_target_image(path, grayscale=False, color_mode='rgb', target_size=(256, 1600),
                      interpolation='nearest'):
    return load_img(path=path, grayscale=grayscale, color_mode=color_mode,
                    target_size=target_size, interpolation=interpolation)


def input_predict_softmax(filename, segs,crop=False):
    load_dir =os.path.join(os.path.join(os.getcwd(),'data'),'train_images/')
    for i in tqdm(range(len(filename))):
        fn = filename[i]
        cur_img = np.asarray(load_target_image(path=load_dir + fn))
        cur_img = gamma_correction(cur_img)
        cur_img = contrast_enhancement(cur_img)
        masks = merge_masks(fn, segs, reshape=None)
        mask_merge = np.sum(masks, axis=2)
        background = np.expand_dims(np.array(mask_merge == 0, dtype=np.int8), axis=-1)
        outmask = np.expand_dims(np.concatenate((masks, background), axis=-1), axis=0)
        if crop:
            img_array = crop_img(cur_img)
        else:
            img_array =np.expand_dims(cv2.resize(cur_img,(512,256)),axis=0)
        if i == 0:
            batch_rgb = img_array
            batch_mask = outmask
        else:
            batch_rgb, batch_mask = np.concatenate((batch_rgb, img_array), axis=0), np.concatenate(
                (batch_mask, outmask), axis=0)
    return batch_rgb, batch_mask

def input_predict_sigmoid(filename, segs,preprocess='resize',normalize=True):
    load_dir =os.path.join(os.path.join(os.getcwd(),'data'),'train_images/')
    for i in tqdm(range(len(filename))):
        fn = filename[i]
        cur_img = np.asarray(load_target_image(path=load_dir + fn))
        cur_img = gamma_correction(cur_img)
        cur_img = contrast_enhancement(cur_img)
        if normalize:
            cur_img=cur_img/255
        outmask = np.expand_dims(merge_masks(fn, segs, reshape=None),axis=0)
        if preprocess=='crop':
            img_array = crop_img(cur_img)
        elif preprocess=='resize':
            img_array =np.expand_dims(cv2.resize(cur_img,(512,256)),axis=0)
        else:
            img_array =np.expand_dims(cur_img,axis=0)

        if i == 0:
            batch_rgb = img_array
            batch_mask = outmask
        else:
            batch_rgb, batch_mask = np.concatenate((batch_rgb, img_array), axis=0), np.concatenate((batch_mask, outmask), axis=0)
    return batch_rgb, batch_mask



class tta_wrapper():
    def __init__(self,model):
        self.model = model
    def predict(self,x,ave='temperture'):
        for i in range(x.shape[0]):
            out1 = self.model.predict(np.expand_dims(F.vflip(x[i,]),axis=0))
            out2 = self.model.predict(np.expand_dims(F.hflip(x[i,]),axis=0))
            out3 = self.model.predict(np.expand_dims(F.vflip(F.hflip(x[i,])),axis=0))
            out4 = self.model.predict(np.expand_dims(x[i,],axis=0))
            ave = (F.vflip(out1)+F.hflip(out2)+F.hflip(F.vflip(out3))+out4)/4
            if i == 0:
                output = ave
            else:
                output =np.concatenate((output,ave),axis=0)
        return output

def model_enssenble_th(models,valid_x,train_seg,threhold,size,tta=False,preprocess='resize'):
    batch_rgb, batch_mask=input_predict_sigmoid(valid_x,train_seg,preprocess)
    pred_list = []
    for model in models:
        if tta:
            model= tta_wrapper(model)
        pred = model.predict(batch_rgb)
        pred_list.append(pred)

    plus_pred = np.zeros(pred.shape)
    for j in range(len(pred_list)):
        plus_pred+=pred_list[j]
    plus_pred= plus_pred/len(pred_list)
    output_mask = np.zeros(plus_pred.shape)
    for i in range(pred.shape[0]):
        predimg=plus_pred[i,:,:,:]
        gt=batch_mask[i,:,:,:]
        for j in range(4):
            mask =np.array(predimg[:,:,j]>threhold[j],np.uint8)
            mask = single_mask_reduce(mask,size[j])
            output_mask[i,:,:,j] = mask
    return output_mask

class DataGenerator_mask(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(256,1600), n_channels=3,
                 n_classes=4, shuffle=False,agument=True,reshape=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = r'J:\kaggle competition\stealdefects\data'
        self.augment = agument
        self.reshape = reshape
        cvs_path =os.path.join(self.path,'train.csv')
        df = pd.read_csv(cvs_path)
        df.fillna('', inplace=True)
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in list_IDs for c in [1,2,3,4]])
        self.df=df
        self.samples=len(self.list_IDs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.augment:
            X,y  = self.__augmentation(X,y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.reshape==None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size, *self.dim, self.n_classes),dtype=int)
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))
            y = np.empty((self.batch_size, *self.reshape, self.n_classes),dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            folder, image_id = list_IDs_temp[i].split('/')
            image = cv2.imread(self.path+'/%s/%s'%(folder,image_id),cv2.IMREAD_COLOR)
            if self.reshape!=None:
                image=np_resize(image,self.reshape)
            image = np.array(image,dtype=np.float32)
            X[i,] = image/255
            rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
            ]
            mask = build_masks(rle,(256,1600),reshape=self.reshape)
            # Store class
            y[i,] = mask

        return X, y

    def __randomtransform(self,img,mask):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=30),albu.GridDistortion(p=0.5)])
        composed = composition(image=img, mask=mask)
        aug_img = composed['image']
        aug_masks = composed['mask']
        return aug_img, aug_masks

    def __augmentation(self,X,y):
        for i in range(X.shape[0]):
            X[i,],y[i,]=self.__randomtransform(X[i,],y[i,])
        return X, y




class DataGenerator_classfication(keras.utils.Sequence):
    def __init__(self, list_IDs,folder, batch_size=32, dim=(256,1600), n_channels=3,
                 n_classes=4, shuffle=True,reshape=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.reshape=reshape
        self.augment= True
        self.path = os.path.join(os.getcwd(),'data')
        self.folder=folder
        cvs_path =os.path.join(self.path,'train.csv')
        df = pd.read_csv(cvs_path)
        df.fillna('', inplace=True)
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in list_IDs for c in [1,2,3,4]])
        self.df=df
        self.samples=len(self.list_IDs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.augment:
            X= self.__augmentation(X)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.reshape==None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels),dtype=np.float32)

        y = np.empty((self.batch_size, self.n_classes),dtype=np.int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image_id = list_IDs_temp[i]
            image = cv2.imread(self.path+'/%s/%s'%(self.folder,image_id),cv2.IMREAD_COLOR)
            if self.reshape!=None:
                image=np_resize(image,self.reshape)
            image=np.array(image,dtype=np.float32)
            X[i,] = image/255
            rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
            ]
            mask = build_masks(rle,(256,1600),reshape=self.reshape)
            label = (mask.reshape(4,-1).sum(1)>500).astype(np.int32)
            # Store class
            y[i,] = label

        return X, y

    def __randomtransform(self,img):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=30)])
        composed = composition(image=img)
        aug_img = composed['image']
        return aug_img

    def __augmentation(self,X):
        for i in range(X.shape[0]):
            X[i,]=self.__randomtransform(X[i,])
        return X


class SGDRScheduler(Callback):
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)

def reconstruct(output):
    reconstruct = np.zeros([int(output.shape[0] / 7), 256, 1600, output.shape[3]], dtype=np.float32)
    mark = np.zeros([int(output.shape[0] / 7), 256, 1600, output.shape[3]], dtype=np.uint8)
    for i in range(int(output.shape[0] / 7)):
        single_out = output[i * 7:(i + 1) * 7, ]
        for j in range(7):
            reconstruct[i, :, j * 200:j * 200 + 400, :] = single_out[j,] + reconstruct[i, :, j * 200:j * 200 + 400, :]
            mark[i, :, j * 200:j * 200 + 400, :] += 1
    reconstruct = np.array(reconstruct / mark, dtype=np.uint8)
    return reconstruct

def predict(unet,valid_x,train_seg,threhold,size,preprocess='original',tta=True):
    d=0
    for idx in range(4):
        predict_ori =valid_x[idx*250:(idx+1)*250]
        batch_rgb, batch_mask=input_predict_sigmoid(predict_ori,train_seg,preprocess)
        if tta:
            ta_unet= tta_wrapper(unet)
            pred=ta_unet.predict(batch_rgb)
        else:
            pred=unet.predict(batch_rgb,verbose=1)
        if preprocess=='crop':
            pred=reconstruct(pred)
        elif preprocess=='resize':
            tempstore= np.zeros([pred.shape[0],256,1600,4])
            for j in range(pred.shape[0]):
                tempstore[j,]=cv2.resize(pred[j,], (1600,256))
            pred= tempstore
        else:
            pred=pred

        for i in range(pred.shape[0]):
            predimg=pred[i,:,:,:]
            gt=batch_mask[i,:,:,:]
            for j in range(4):
                mask =np.array(predimg[:,:,j]>threhold[j],np.uint8)
                mask = single_mask_reduce(mask,size[j])
                if mask.sum()==0 and np.sum(gt[:,:,j])==0:
                    d+=1
                else:
                    d+=Dice(gt[:,:,j],mask)
    return d
