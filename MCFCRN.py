import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Bidirectional, LSTM, Flatten
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Activation, LayerNormalization
from tensorflow.keras.layers import Cropping2D, Add, Concatenate, Reshape, Lambda
from tensorflow.keras.regularizers import L2

print(tf.config.list_physical_devices("GPU"))

win_len = 78
def extract_patches(img, img_height, stride=1):
    return tf.image.extract_patches(img, sizes=[1, img_height, win_len, 1], 
                                    strides=[1, 1, stride, 1], rates=[1, 1, 1, 1], 
                                    padding='VALID')

#%%
class BaseNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 gamma_initializer="ones",
                 beta_initializer="zeros",
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        if isinstance(axis, int):
            axis = [axis]
        self.axis = list(axis)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)

    def build(self, input_shape):
        self.axis = [len(input_shape) + ax if ax < 0 else ax for ax in self.axis]
        self.shape = tuple(input_shape[ax] for ax in self.axis)
        self.para_shape = tuple(input_shape[ax] if ax in self.axis else 1 for ax in range(1, len(input_shape)))
    
    def build_weight(self, add_name=""):
        gamma, beta = 1., 0.
        if self.scale:
            gamma = self.add_weight(
                name=f'gamma{add_name}',
                shape=self.shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                trainable=True
            )
        if self.center:
            beta = self.add_weight(
                name=f'beta{add_name}',
                shape=self.shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                trainable=True
            )
        
        return gamma, beta

    @tf.function
    def _get_normalize(self, inputs, training=None):        
        mean = tf.math.reduce_mean(inputs, axis=self.reduce_axis, keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=self.reduce_axis, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= tf.reshape(self.gamma, self.para_shape)
        if self.center:
            outputs += tf.reshape(self.beta, self.para_shape)
        return outputs

class InstanceNormalization(BaseNormalization):
    '''
        axis: at which channel locate; different from LayerNormalization
        compute statistics across (Height, Width) ONLY
        e.g. input_shape = (Batch, Height, Width, Channel); axis=-1
        e.g. input_shape = (Batch, Time, Frequency, Channel); axis=-1
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def build(self, input_shape):
        super().build(input_shape)
        self.gamma, self.beta = self.build_weight()
        self.reduce_axis = [ax for ax in range(1, len(input_shape)) if ax not in self.axis]

    @tf.function
    def call(self, inputs, training=None):
        outputs = self._get_normalize(inputs, training)
        
        return outputs

#%%
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Add total CTC loss via `self.add_loss()`
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        # reduce batch_len x 1 loss matrix to scalar for regularizer
        self.add_loss(tf.keras.backend.sum(loss))

        return y_pred

#%%
def MCFCRN(total_words, input_shape, print_summary=False):
    img = tf.keras.Input(name="images", shape=input_shape, dtype="float32")
    labels = tf.keras.Input(name="label", shape=(None,), dtype="float32")
    
    img_height, _, img_channel = input_shape
    args = {"img_height": img_height, "stride": 20}
    patches = Lambda(extract_patches, arguments=args)(img)
    
    x3 = Reshape((-1, img_height, win_len, img_channel))(patches)
    x2 = TimeDistributed(Cropping2D(cropping=((0, 0), (8, 8))))(x3)
    x1 = TimeDistributed(Cropping2D(cropping=((0, 0), (8, 8))))(x2)
    
    
    '''
        x1
    '''
    #x1 = tf.pad(x1, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation="relu", 
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x1)
    
    #x1 = tf.pad(x1, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x1 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", 
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x1)
    
    #x1 = tf.pad(x1, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x1 = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), 
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(InstanceNormalization())(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    #x1 = tf.pad(x1, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x1 = TimeDistributed(Conv2D(128, (1, 1), strides=(1, 1), 
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(InstanceNormalization())(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    x1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x1)
    
    #x1 = tf.pad(x1, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x1 = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(InstanceNormalization())(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    
    '''
        x2
    '''
    #x2 = tf.pad(x2, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation="relu",
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x2)
    
    #x2 = tf.pad(x2, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x2 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation="relu",
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x2)
    
    #x2 = tf.pad(x2, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x2 = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(InstanceNormalization())(x2)
    x2 = TimeDistributed(Activation("relu"))(x2)
    #x2 = tf.pad(x2, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x2 = TimeDistributed(Conv2D(128, (1, 1), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(InstanceNormalization())(x2)
    x2 = TimeDistributed(Activation("relu"))(x2)
    x2 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x2)
    
    #x2 = tf.pad(x2, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x2 = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(InstanceNormalization())(x2)
    x2 = TimeDistributed(Activation("relu"))(x2)
    
    '''
        x3
    '''
    #x3 = tf.pad(x3, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x3 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation="relu",
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x3)
    
    #x3 = tf.pad(x3, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x3 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation="relu",
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x3)
    
    #x3 = tf.pad(x3, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x3 = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), activation="relu",
                                kernel_regularizer=L2(0.001)))(x3)
    #x3 = tf.pad(x3, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x3 = TimeDistributed(Conv2D(128, (1, 1), strides=(1, 1), activation="relu",
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x3)
    
    #x3 = tf.pad(x3, ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    x3 = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1), activation="relu",
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(InstanceNormalization())(x3)
    x3 = TimeDistributed(Activation("relu"))(x3)
    
    # FCRN
    x1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x1)
    x1 = TimeDistributed(Conv2D(512, (1, 1), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(InstanceNormalization())(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    x1 = TimeDistributed(Conv2D(512, (3, 1), strides=(3, 1),
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(InstanceNormalization())(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    x1 = TimeDistributed(Conv2D(1024, (2, 1), strides=(2, 1),
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = TimeDistributed(InstanceNormalization())(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    x1 = TimeDistributed(Flatten())(x1)
    
    x1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x1)
    #x1 = Add()([x1, y1])
    y1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x1)
    x1 = Add()([x1, y1])
    y1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x1)
    x1 = Add()([x1, y1])
    
    # 2-FCRN
    x2 = tf.pad(x2, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
    x2 = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(InstanceNormalization())(x2)
    x2 = TimeDistributed(Activation("relu"))(x2)
    x2 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x2)
    x2 = TimeDistributed(Conv2D(512, (1, 1), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(InstanceNormalization())(x2)
    x2 = TimeDistributed(Activation("relu"))(x2)
    x2 = TimeDistributed(Conv2D(512, (3, 1), strides=(3, 1),
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(InstanceNormalization())(x2)
    x2 = TimeDistributed(Activation("relu"))(x2)
    x2 = TimeDistributed(Conv2D(1024, (2, 1), strides=(2, 1),
                                kernel_regularizer=L2(0.001)))(x2)
    x2 = TimeDistributed(InstanceNormalization())(x2)
    x2 = TimeDistributed(Activation("relu"))(x2)
    x2 = TimeDistributed(Flatten())(x2)
    
    x2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x2)
    #x2 = Add()([x2, y2])
    y2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x2)
    x2 = Add()([x2, y2])
    y2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x2)
    x2 = Add()([x2, y2])
    
    # 3-FCRN
    x3 = tf.pad(x3, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
    x3 = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(InstanceNormalization())(x3)
    x3 = TimeDistributed(Activation("relu"))(x3)
    x3 = tf.pad(x3, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
    x3 = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(InstanceNormalization())(x3)
    x3 = TimeDistributed(Activation("relu"))(x3)
    x3 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x3)
    x3 = TimeDistributed(Conv2D(512, (1, 1), strides=(1, 1),
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(InstanceNormalization())(x3)
    x3 = TimeDistributed(Activation("relu"))(x3)
    x3 = TimeDistributed(Conv2D(512, (3, 1), strides=(3, 1),
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(InstanceNormalization())(x3)
    x3 = TimeDistributed(Activation("relu"))(x3)
    x3 = TimeDistributed(Conv2D(1024, (2, 1), strides=(2, 1),
                                kernel_regularizer=L2(0.001)))(x3)
    x3 = TimeDistributed(InstanceNormalization())(x3)
    x3 = TimeDistributed(Activation("relu"))(x3)
    x3 = TimeDistributed(Flatten())(x3)
    
    x3 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x3)
    #x3 = Add()([x3, y3])
    y3 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x3)
    x3 = Add()([x3, y3])
    y3 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                            kernel_regularizer=L2(0.001)), merge_mode='sum')(x3)
    x3 = Add()([x3, y3])
    
    x1 = Dense(512, activation="relu", kernel_regularizer=L2(0.001))(x1)
    x1 = Dense(512, activation="relu", kernel_regularizer=L2(0.001))(x1)
    
    x2 = Dense(1024, activation="relu", kernel_regularizer=L2(0.001))(x2)
    x2 = Dense(1024, activation="relu", kernel_regularizer=L2(0.001))(x2)
    
    x3 = Dense(1536, activation="relu", kernel_regularizer=L2(0.001))(x3)
    x3 = Dense(1536, activation="relu", kernel_regularizer=L2(0.001))(x3)

    x = Concatenate(name="concat")([x1, x2, x3])
    x = Dense(total_words + 1, activation="softmax", name="soft_out", 
              kernel_regularizer=L2(0.001))(x)
    output = CTCLayer(name="ctc_loss")(labels, x)
    
    model = tf.keras.Model(inputs=[img, labels], outputs=output, name="MCFCRN")
    if print_summary:
        model.summary()
    return model

def transfer_learning(base_model, total_words, input_shape, print_summary=False):
    img = tf.keras.Input(name="images", shape=input_shape, dtype="float32")
    labels = tf.keras.Input(name="label", shape=(None,), dtype="float32")
    
    base_model.trainable = False
    x = base_model(img)
    x = Dense(1024, use_bias=False)(x) # Embedding layer with float inputs
    #x = Embedding(1024, 512, trainable=True)(x)
    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                           kernel_regularizer=L2(0.001)), merge_mode='sum')(x)
    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3,
                           kernel_regularizer=L2(0.001)), merge_mode='sum')(x)
    x = Dense(1024, activation="relu", kernel_regularizer=L2(0.001))(x)
    x = Dense(1024, activation="relu", kernel_regularizer=L2(0.001))(x)
    
    x = Dense(total_words+1, activation="softmax", name="soft_out",
              kernel_regularizer=L2(0.001))(x)
    output = CTCLayer(name="ctc_loss")(labels, x)
    
    model = tf.keras.Model(inputs=[img, labels], outputs=output, name="MCFCRN")
    if print_summary:
        model.summary()
    return model

#%%
def create_dataset(feature, labels):
    '''
        "images" and "label" should match the name of the tf model two inputs
    '''
    return {"images":feature, "label":labels}, labels

#%%
from tensorflow.keras.optimizers.legacy import Adadelta
from tensorflow.keras.optimizers import Adam
# Adadelta encounters gradient vanishing problem; probably because of small mini-batch

batch_size = 36 # max at 8 for K80:4; 36 for V100:3

from Get_Data import Preprocess

train_path = "WPTT2.0-Train"
PRE = Preprocess(path=train_path)
#%%
train_data = tf.data.Dataset.from_generator(PRE.read_path, (tf.float32, tf.float32), 
                                         args=(train_path,))
train_data = train_data.map(create_dataset, num_parallel_calls=tf.data.AUTOTUNE)
# setting large batch size even like 2 raise out of GPU memory error, so probably
# using InstanceNormalization instead of TimeDistributed(BatchNormalization())
train_data = train_data.shuffle(buffer_size=PRE.total_entries//100).batch(batch_size, drop_remainder=True,
    num_parallel_calls=tf.data.AUTOTUNE)
print(PRE.total_entries)

#%%
# initial_learning_rate * decay_rate ^ (step / decay_steps)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

#%%
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    MCFCRN_model = MCFCRN(len(PRE.characters), PRE.input_shape)
    MCFCRN_model.compile(optimizer=Adam(learning_rate=lr_schedule))
    #MCFCRN_model.compile(optimizer=Adadelta(learning_rate=0.0001, rho=0.9))

MCFCRN_model.fit(train_data, epochs=200, shuffle=True)
MCFCRN_model.save("MCFCRN")
#%%
test_path = "WPTT2.0-Test"
test_data = tf.data.Dataset.from_generator(PRE.read_path, (tf.float32, tf.float32), 
                                           args=(test_path,)).batch(1)

MCFCRN_prediction_model = tf.keras.models.Model(
    MCFCRN_model.get_layer(name="images").input, MCFCRN_model.get_layer(name="soft_out").output
)
y_MCFCRN = MCFCRN_prediction_model.predict(test_data)
print(PRE.model_decoder(y_MCFCRN))

#%%
for f, label in test_data:
    print(PRE.label_decoder(label.numpy()))
    
#%%
base_model = tf.keras.models.Model(
    MCFCRN_model.get_layer(name="images").input, MCFCRN_model.get_layer(name="concat").output
)

with strategy.scope():
    transfer_model = transfer_learning(base_model, len(PRE.characters), PRE.input_shape)
    transfer_model.compile(optimizer=Adam(learning_rate=lr_schedule))
    #transfer_model.compile(optimizer=Adadelta(learning_rate=0.0001, rho=0.9))

transfer_model.fit(train_data, epochs=200, shuffle=True)
transfer_model.save("Transfer")

#%%
transfer_prediction_model = tf.keras.models.Model(
    transfer_model.get_layer(name="images").input, transfer_model.get_layer(name="soft_out").output
)

y_transfer = transfer_prediction_model.predict(test_data)
print(PRE.model_decoder(y_transfer))
