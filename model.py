from blocks import *
import efficientnet.keras as efn
from keras.layers import Input, MaxPool2D, Flatten
# effcient net unet

def effcient0_unet(input_shape,filters =512,out_dim=4,act="sigmoid"):
    model = efn.EfficientNetB0(include_top=False,input_shape=input_shape,weights=None)
    blockid = ['block2a_expand_activation','block3a_expand_activation','block4a_expand_activation','block6a_expand_activation','block7a_project_bn']
    block_outputs = []
    for i in range(len(blockid)):
        blockout = model.get_layer(blockid[i]).output
        block_outputs.append(blockout)

    input = model.input
    inputc = Conv2D(16, (3,3), padding="same", activation="relu")(input)
    x = block_outputs.pop()
    for i in range(len(blockid)-1):
        x= transconv_block(x,int(filters/(2**i)),3,2,skip=block_outputs.pop())
    x= transconv_block(x,16,3,2,skip=inputc)
    output_layer = Conv2D(out_dim, (1,1), padding="same", activation=act)(x)
    model = Model(input, output_layer)
    return model

def effcient1_unet(input_shape,filters =512,out_dim=4,act="sigmoid"):
    model = efn.EfficientNetB1(include_top=False,input_shape=input_shape,weights=None)
    blockid = ['block2a_expand_activation','block3a_expand_activation','block4a_expand_activation','block6a_expand_activation','block7b_add']
    block_outputs = []
    for i in range(len(blockid)):
        blockout = model.get_layer(blockid[i]).output
        block_outputs.append(blockout)

    input = model.input
    inputc = Conv2D(16, (3,3), padding="same", activation="relu")(input)
    x = block_outputs.pop()
    for i in range(len(blockid)-1):
        x= transconv_block(x,int(filters/(2**i)),3,2,skip=block_outputs.pop())
    x= transconv_block(x,16,3,2,skip=inputc)
    output_layer = Conv2D(out_dim, (1,1), padding="same", activation=act)(x)
    model = Model(input, output_layer)
    return model


def squeeze_excitation_unet(input_shape,filters=512,out_dim=4,act="sigmoid"):
    inputs =Input(input_shape)
    x =  Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    d1,x = downsample_block(x,int(filters/16), 3, 1, padding='same', activation=True,bottom=False,SE=True)
    d2,x= downsample_block(x,int(filters/8), 3, 1, padding='same', activation=True,bottom=False,SE=True)
    d3,x= downsample_block(x,int(filters/4), 3, 1, padding='same', activation=True,bottom=False,SE=True)
    d4,x= downsample_block(x,int(filters/2), 3, 1, padding='same', activation=True,bottom=False,SE=True)
    x= downsample_block(x,int(filters), 3, 1, padding='same', activation=True,bottom=True,SE=True)

    x= transconv_block(x,int(filters/2),3,2,skip=d4)
    x= transconv_block(x,int(filters/4),3,2,skip=d3)
    x= transconv_block(x,int(filters/8),3,2,skip=d2)
    x= transconv_block(x,int(filters/16),3,2,skip=d1)
    x =  Conv2D(out_dim, (1,1), padding="same", activation=act)(x)
    model = Model(inputs, x)
    return model

def HRnet(input_shape,C=32,out_dim=4,act="sigmoid",expand_filter=256):
    inputs =Input(input_shape)
    hx,lx = branch_bottle(inputs,expand_filter,C)
    hx = branch_base(hx,C)
    lx = branch_base(lx,int(C*2))

    hx,lx,llx = fusion1(hx,lx,C)
    hx = branch_base(hx,C)
    lx = branch_base(lx,int(C*2))
    llx= branch_base(llx,int(C*4))

    hx,lx,llx,lllx =fusion2(hx,lx,llx,C)
    hx = branch_base(hx,C)
    lx = branch_base(lx,int(C*2))
    llx= branch_base(llx,int(C*4))
    lllx=branch_base(lllx,int(C*6))

    hx,lx,llx,lllx =fusion3(hx,lx,llx,lllx,C)
    x =final_layer(hx,lx,llx,lllx)
    out =  Conv2D(out_dim, (1,1), padding="same", activation=act)(x)
    model= Model(inputs, out)
    return model

def resnet(input_shape,type='cir',filter_num=64):
    input=Input(input_shape)
    x =Conv2D(64,kernel_size=7,strides=2,padding='same',name='conv1')(input)
    x =BatchNormalization(axis=-1, epsilon=1.1e-5,name='bn1')(x)
    x =Activation('relu',name='act1')(x)
    x = MaxPool2D(pool_size=2,padding='same',name='maxp1')(x)
    for num_block in range(5):
        for res_block in range(3):
            id_num = num_block*3+res_block
            if num_block>0 and res_block==0:
                x =resnet_layer(x,id_num,filter_num,2)
            else:
                x =resnet_layer(x,id_num,filter_num)
        if num_block >0:
            filter_num =filter_num*2
    x = GlobalAveragePooling2D(name='GlobalAveragePooling2D')(x)
    #x = Flatten(name='flatten_layer')(x)
    if type=='cir':
        x=Dense(10, activation='softmax',kernel_initializer='he_normal',name='cir')(x)
    else:
        x=Dense(4, activation='sigmoid',kernel_initializer='he_normal',name='steel')(x)
    model = Model(inputs=input, outputs=x)
    return model
