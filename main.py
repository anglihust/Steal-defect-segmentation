from train import *
from glob import glob
from uilt import *
from loss import *
cpath = os.getcwd()
model_save_folder = os.path.join(cpath,'model_weight')
main_dir = os.path.join(cpath,'data')
cl_model_path= os.path.join(cpath,'cl_net.h5')

def main(config):
    if config.is_train_cl:
        cl_net = train_resent32(config)
        cl_net.train_steel_data()
    if config.is_train_seg:
        seg_net = train_segmentation_net(config)
        seg_net.train_model()

    if config.is_inferece:
        train_fns = os.listdir(os.path.join(main_dir, 'train_images'))
        test_fns = os.listdir(os.path.join(main_dir,'test_images'))
        train_seg = pd.read_csv(os.path.join(main_dir,'train.csv'))
        train_seg['ImageId'] = train_seg['ImageId_ClassId'].map(lambda x : x.split('_')[0])
        train_seg['ClassId'] = train_seg['ImageId_ClassId'].map(lambda x : x.split('_')[1])
        train_seg = train_seg.drop(['ImageId_ClassId'], axis = 1)
        train_seg['has_label'] = train_seg['EncodedPixels'].map(lambda x : 1 if isinstance(x,str) else 0)
        valid_x=read('valid_b1_1000.npy')
        # using grid search to find best threshold and min area
        threhold=[0.5,0.5,0.5,0.5]
        size=[1600,1600,3200,3200]

        model_net= []
        model_path=[]
        for ext in('/*.h5'):
            model_path.extend(glob(model_save_folder+ext))
        #load models in model_save_folder
        for i in range(len(model_path)):
            net = keras.models.load_model(model_path[i],custom_objects={'Dice': Dice, 'Dice_coef': Dice_coef})
            model_net.append(net)
        # dice for each models
        d_all=np.zeros(len(model_net))
        for i in range(len(model_net)):
            d=predict(model_net[i],valid_x,train_seg,threhold,size,preprocess='resize',tta=True)
            d_all[i]=d

        output = model_enssenble_th(model_net,valid_x,train_seg,threhold,size,tta=True,preprocess='resize')
        if config.cl_clean:
            cl_model = keras.models.load_model(os.path.join(os.getcwd(),'cl_best_weight.hdf5'))
            batch_rgb, batch_mask=input_predict_sigmoid(valid_x,train_seg,preprocess='resize')
            cl_out = cl_model.predict(batch_rgb)
            for i in range(cl_out.shape[0]):
                for j in range(4):
                    output[i,:,:,j] = output[i,:,:,j] *cl_out[j,i]


