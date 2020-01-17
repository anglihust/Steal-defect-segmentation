from keras.layers import concatenate, Concatenate,UpSampling2D,LeakyReLU, MaxPooling2D,Dropout,Conv2D, Add, Conv2DTranspose,GlobalAveragePooling2D,Dense,Activation,multiply
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers, constraints, regularizers, layers
import keras
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.activations import sigmoid


def swish(x):
    return x*sigmoid(x)

def conv_block(x, filters, size, strides=1, padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def transconv_block(x,filters, size, strides=2, padding='same', activation=True,skip=None,SE=False):
    x= Conv2DTranspose(filters, size, strides=strides, padding=padding)(x)
    if skip is not None:
        x=concatenate([x, skip])
    x = conv_block(x, filters, size)
    if SE:
        x= SE_block(x)
    x = conv_block(x, filters, size, activation=activation)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = conv_block(x, num_filters, 3)
    x = conv_block(x, num_filters, 3, activation=False)
    x = Add()([x, blockInput])
    return x

def SE_block(inputs,se_ratio=8):
    input_channels = inputs._keras_shape[-1]
    reduced_channels = max(input_channels // se_ratio, 8)
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(units=reduced_channels, kernel_initializer="he_normal")(x)
    x = Activation('relu')(x)
    x = Dense(units=input_channels, activation='sigmoid', kernel_initializer="he_normal")(x)
    return multiply([inputs, x])

def downsample_block(x,filters, size, strides=1, padding='same', activation=True,bottom=False,SE=False):
    x = conv_block(x, filters, size,strides,padding=padding)
    if SE:
        x= SE_block(x)
    x = conv_block(x, filters, size, strides,padding=padding,activation=activation)
    if bottom:
        return x
    else:
        return (x,MaxPooling2D(pool_size=2)(x))

#  HRnet blocks
def basic_Block(input, filters, size= 3,strides=1, shortcut=False):
    x = conv_block(input, filters, size, strides=1, padding='same', activation=True)
    x = Conv2D(filters, size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if shortcut:
        residual = Conv2D(filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = Add()([x, residual])
    else:
        x = Add()([x, input])
    x = Activation('relu')(x)
    return x

def bottleneck_Block(input, filters , strides=(1, 1), shortcut=False):
    expansion = 4
    de_filters = int(filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if shortcut:
        residual = Conv2D(filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = Add()([x, residual])
    else:
        x = Add()([x, input])

    x = Activation('relu')(x)
    return x

def branch_bottle(x,filters=256,C=32):
    x = conv_block(x, filters, 3, strides=1, padding='same', activation=True)
    x = bottleneck_Block(x, filters, shortcut=True)
    x = bottleneck_Block(x, filters, shortcut=False)
    x = bottleneck_Block(x, filters, shortcut=False)
    x = bottleneck_Block(x, filters, shortcut=False)
    hx = conv_block(x, int(C), 3, strides=1, padding='same', activation=True)
    lx = conv_block(x, int(C*2), 3, strides=2, padding='same', activation=True)
    return (hx,lx)

def branch_base(x,C =32,shortcut=False):
    x = basic_Block(x, C, shortcut=shortcut)
    x = basic_Block(x, C, shortcut=shortcut)
    x = basic_Block(x, C, shortcut=shortcut)
    x = basic_Block(x, C, shortcut=shortcut)
    return x

# def branch2_block1(x,C =64,shortcut=False):
#     x = basic_Block(x, C, shortcut=shortcut)
#     x = basic_Block(x, C, shortcut=shortcut)
#     x = basic_Block(x, C, shortcut=shortcut)
#     x = basic_Block(x, C, shortcut=shortcut)
#     return x

def fusion1(hx,lx,C=32):
    llx1 = Conv2D(int(C*4), 3, strides=2, padding='same',activation=None)(lx)
    llx2 = Conv2D(int(C), 3, strides=2, padding='same',activation=None)(hx)
    llx2 =  Conv2D(int(C*4), 3, strides=2, padding='same',activation=None)(llx2)
    llx_o  = Add()([llx1, llx2])

    hx2 = UpSampling2D(size=2)(lx)
    hx2 =Conv2D(int(C), 1, strides=1, padding='same',activation=None)(hx2)
    hx_o= Add()([hx, hx2])

    lx2=Conv2D(int(C*2), 3, strides=2, padding='same',activation=None)(hx)
    lx_o= Add()([lx, lx2])
    return (hx_o,lx_o,llx_o)

def fusion2(hx,lx,llx,C=32):
    lllx1 =  Conv2D(int(C*6), 3, strides=2, padding='same',activation=None)(llx)
    lllx2 = Conv2D(int(C*2), 3, strides=2, padding='same',activation=None)(lx)
    lllx2 =  Conv2D(int(C*6), 3, strides=2, padding='same',activation=None)(lllx2)
    lllx3 = Conv2D(int(C), 3, strides=2, padding='same',activation=None)(hx)
    lllx3 =  Conv2D(int(C), 3, strides=2, padding='same',activation=None)(lllx3)
    lllx3 =  Conv2D(int(C*6), 3, strides=2, padding='same',activation=None)(lllx3)
    lllx_o = Add()([lllx1,lllx2,lllx3])

    hx1 = UpSampling2D(size=2)(lx)
    hx1 =Conv2D(int(C), 1, strides=1, padding='same',activation=None)(hx1)
    hx2 =UpSampling2D(size=4)(llx)
    hx2 =Conv2D(int(C), 1, strides=1, padding='same',activation=None)(hx2)
    hx_o = Add()([hx1,hx2,hx])

    lx1 = Conv2D(int(2*C), 3, strides=2, padding='same',activation=None)(hx)
    lx2 = UpSampling2D(size=2)(llx)
    lx2 =Conv2D(int(2*C), 1, strides=1, padding='same',activation=None)(lx2)
    lx_o = Add()([lx1,lx2,lx])

    llx1 = Conv2D(int(C), 3, strides=2, padding='same',activation=None)(hx)
    llx1 = Conv2D(int(4*C), 3, strides=2, padding='same',activation=None)(llx1)
    llx2 = Conv2D(int(4*C), 3, strides=2, padding='same',activation=None)(lx)
    llx_o = Add()([llx1,llx2,llx])
    return (hx_o,lx_o,llx_o,lllx_o)

def fusion3(hx,lx,llx,lllx,C):
    lllx1 =  Conv2D(int(C*6), 3, strides=2, padding='same',activation=None)(llx)
    lllx2 = Conv2D(int(C*2), 3, strides=2, padding='same',activation=None)(lx)
    lllx2 =  Conv2D(int(C*6), 3, strides=2, padding='same',activation=None)(lllx2)
    lllx3 = Conv2D(int(C), 3, strides=2, padding='same',activation=None)(hx)
    lllx3 =  Conv2D(int(C), 3, strides=2, padding='same',activation=None)(lllx3)
    lllx3 =  Conv2D(int(C*6), 3, strides=2, padding='same',activation=None)(lllx3)
    lllx_o = Add()([lllx1,lllx2,lllx3,lllx])

    hx1 = UpSampling2D(size=2)(lx)
    hx1 =Conv2D(int(C), 1, strides=1, padding='same',activation=None)(hx1)
    hx2 =UpSampling2D(size=4)(llx)
    hx2 =Conv2D(int(C), 1, strides=1, padding='same',activation=None)(hx2)
    hx3 =UpSampling2D(size=8)(lllx)
    hx3 =Conv2D(int(C), 1, strides=1, padding='same',activation=None)(hx3)
    hx_o = Add()([hx1,hx2,hx3,hx])

    lx1 = Conv2D(int(2*C), 3, strides=2, padding='same',activation=None)(hx)
    lx2 = UpSampling2D(size=2)(llx)
    lx2 =Conv2D(int(2*C), 1, strides=1, padding='same',activation=None)(lx2)
    lx3 = UpSampling2D(size=4)(lllx)
    lx3 =Conv2D(int(2*C), 1, strides=1, padding='same',activation=None)(lx3)
    lx_o = Add()([lx1,lx2,lx3,lx])

    llx1 = Conv2D(int(C), 3, strides=2, padding='same',activation=None)(hx)
    llx1 = Conv2D(int(4*C), 3, strides=2, padding='same',activation=None)(llx1)
    llx2 = Conv2D(int(4*C), 3, strides=2, padding='same',activation=None)(lx)
    llx3 = UpSampling2D(size=2)(lllx)
    llx3 =Conv2D(int(4*C), 1, strides=1, padding='same',activation=None)(llx3)
    llx_o= Add()([llx1,llx2,llx3,llx])


    return (hx_o,lx_o,llx_o,lllx_o)

def final_layer(hx,lx,llx,lllx):
    lx = UpSampling2D(size=2)(lx)
    llx= UpSampling2D(size=4)(llx)
    lllx= UpSampling2D(size=8)(lllx)
    out =Concatenate()([hx,lx,llx,lllx])
    return out

#resenet classification blocks
def resnet_layer(inputs,blockid,
                 filternum=16,
                 strides=1):
    conv1 = BatchNormalization(axis=-1, epsilon=1.1e-5,name='bn1'+str(blockid))(inputs)
    conv1 = Activation('relu',name='act1'+str(blockid))(conv1)
    conv1 = Conv2D(filternum,kernel_size=3,strides=strides,padding='same',activation=None,name='conv1'+str(blockid))(conv1)
    conv2 = BatchNormalization(axis=-1, epsilon=1.1e-5,name='bn2'+str(blockid))(conv1)
    conv2  = Activation('relu',name='act2'+str(blockid))(conv2)
    conv2 = Conv2D(filternum,kernel_size=3,strides=1,padding='same',activation=None,name='conv2'+str(blockid))(conv2)
    if inputs[3] != filternum:
        shortcut = Conv2D(filternum, (1, 1),strides=strides, padding='same',name='conv3'+str(blockid))(inputs)
    else:
        shortcut=inputs
    output= Add()([shortcut,conv2])
    return output

class GroupNormalization(keras.layers.Layer):
    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


