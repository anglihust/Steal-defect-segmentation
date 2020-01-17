import os
from keras.datasets import cifar10
from model import *
from uilt import *
from loss import *
from keras.optimizers import Adam
import albumentations as albu
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE , GridDistortion,GaussNoise)
#list of images
train_x=read('train_b1_11568.npy')
valid_x=read('valid_b1_1000.npy')



def lr_schedule(epoch):
    lr = 5e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# train a classfication network to determine the defects type of images
# then the classfication results is used to remove false positive segmentation for segmentation network
class train_resent32():
    def __init__(self,config):
        self.pretrain = config.pretrain
        self.subtract_pixel_mean=1
        self.CIFAR10save_dir = os.path.join(os.path.join(os.getcwd(),'model_weight'),'CIFAR10_best_weight.h5')
        self.steelsave_dir=os.path.join(os.getcwd(),'cl_best_weight.h5')
        self.load_pretrain= False #config.pretrain
        self.batch_size = config.steelbatch

    def train_cifar(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        input_shape = x_train.shape[1:]
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        if self.subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        model=resnet(input_shape,'cir',filter_num=64)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.summary()
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        checkpoint = keras.callbacks.ModelCheckpoint(filepath =self.CIFAR10save_dir,monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True)
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
        lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),validation_data=(x_test, y_test),epochs=200, verbose=1,callbacks=callbacks)

    def train_steel_data(self):
        optimizer= Adam(0.0005)
        res=resnet((256,1600,3),type='steel',filter_num=64)
        res.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        train_id = read('train_b1_11568.npy')
        train_generator=DataGenerator_classfication(train_id,'train_images',batch_size=self.batch_size, dim=(256,1600), n_channels=3,n_classes=4, shuffle=False,reshape=None)
        validate_id = read('valid_b1_1000.npy')
        validation_generator=DataGenerator_classfication(validate_id,'test_images', batch_size=self.batch_size, dim=(256,1600), n_channels=3,n_classes=4, shuffle=False,reshape=None)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath =self.steelsave_dir,monitor='val_acc',verbose=1,save_best_only=True)
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
        lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        history = res.fit_generator(train_generator,steps_per_epoch=train_generator.samples/train_generator.batch_size ,epochs=100,validation_data=validation_generator,validation_steps=validation_generator.samples/validation_generator.batch_size,verbose=1,callbacks=callbacks)
        plot_history(history)


# train segmentation networks
class train_segmentation_net():
    def __init__(self,config):
        self.main_dir = os.path.join(os.path.join(os.getcwd(),'data'))
        self.model_name = config.model_name
        self.steelsave_dir=os.path.join(os.path.join(os.getcwd(),'model_weight'),config.model_name+'.h5')
        self.load_pretrain= False #config.pretrain
        self.batch_size = config.segbatch
        if config.reshape:
            self.input_shape = [256,512,3]
        else:
            self.input_shape = [256,1600,3]

    def build_model(self):
        if self.model_name =='effcient0_unet':
            self.model = effcient0_unet(self.input_shape,filters =512,out_dim=4,act="sigmoid")
        elif self.model_name == 'effcient1_unet':
            self.model = effcient1_unet(self.input_shape,filters =512,out_dim=4,act="sigmoid")
        elif self.model_name == 'SE_unet':
            self.model = squeeze_excitation_unet(self.input_shape,filters=512,out_dim=4,act="sigmoid")
        else:
            self.model = HRnet(self.input_shape,C=16,out_dim=4,act="sigmoid")

    def _data_load(self):
        train_fns = os.listdir(os.path.join(self.main_dir, 'train_images'))
        test_fns = os.listdir(os.path.join(self.main_dir,'test_images'))
        train_seg = pd.read_csv(os.path.join(self.main_dir,'train.csv'))
        train_seg['ImageId'] = train_seg['ImageId_ClassId'].map(lambda x : x.split('_')[0])
        train_seg['ClassId'] = train_seg['ImageId_ClassId'].map(lambda x : x.split('_')[1])
        train_seg = train_seg.drop(['ImageId_ClassId'], axis = 1)
        train_seg['has_label'] = train_seg['EncodedPixels'].map(lambda x : 1 if isinstance(x,str) else 0)
        return train_seg

    def train_model(self):
        self.build_model()
        self.model.summary()
        train_seg = self._data_load()
        aug_for_train = albu.Compose([HorizontalFlip(p=0.5),
                             ShiftScaleRotate(scale_limit=0.2, shift_limit=0.1, rotate_limit=15, p=0.5),
                             GridDistortion(p=0.5)])
        aug_for_valid = albu.Compose([])
        reshape_mask=(self.input_shape[0],self.input_shape[1])
        log_dir =  os.path.join(self.main_dir,'train_images')

        train_aug_gen = input_gen(train_x, train_seg, aug_for_train,log_dir, batch_size = self.batch_size,reshape = reshape_mask)
        valid_aug_gen = input_gen(valid_x, train_seg, aug_for_valid,log_dir, batch_size = self.batch_size,reshape = reshape_mask)
        optimizer = Adam(lr=0.001)
        steps_per_epoch = len(train_x) // self.batch_size
        validation_steps = len(valid_x) // self.batch_size
        self.model.compile(loss = dice_loss, optimizer = optimizer, metrics = [Dice_coef])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'min', factor = 0.5, verbose = 1)
        cp = keras.callbacks.ModelCheckpoint(self.steelsave_dir, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
        es = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min')
        training_callbacks = [reduce_lr, cp, es]
        history = self.model.fit_generator(train_aug_gen, steps_per_epoch = steps_per_epoch, epochs =60,
                              validation_data = valid_aug_gen, validation_steps = validation_steps, verbose = 1, callbacks = training_callbacks)

        plot_history(history)
