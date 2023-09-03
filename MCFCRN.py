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
from tensorflow.keras.layers import GroupNormalization

class InstanceNormalization(GroupNormalization):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(groups=-1, axis=axis, **kwargs)

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
    x1 = InstanceNormalization()(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    x1 = TimeDistributed(Conv2D(512, (3, 1), strides=(3, 1),
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = InstanceNormalization()(x1)
    x1 = TimeDistributed(Activation("relu"))(x1)
    x1 = TimeDistributed(Conv2D(1024, (2, 1), strides=(2, 1),
                                kernel_regularizer=L2(0.001)))(x1)
    x1 = InstanceNormalization()(x1)
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

batch_size = 36 # max at 8 for K80:4; 24 for V100:3

from Get_Data import Preprocess

train_path = "WPTT2.0-Train"
PRE = Preprocess(path=train_path)

#%%
train_data = tf.data.Dataset.from_generator(PRE.read_path, (tf.float32, tf.float32), 
                                         args=(train_path,))
train_data = train_data.map(create_dataset, num_parallel_calls=tf.data.AUTOTUNE)
# setting large batch size even like 2 raise out of GPU memory error, so probably
# using InstanceNormalization instead of TimeDistributed(BatchNormalization())
train_data = train_data.shuffle(buffer_size=PRE.total_entries//100).batch(batch_size)

#%%
test_path = "WPTT2.0-Test"
test_data = tf.data.Dataset.from_generator(PRE.read_path, (tf.float32, tf.string), 
                                           args=(test_path,False)).batch(1)

#%%
# initial_learning_rate * decay_rate ^ (step / decay_steps)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

#%%
from datetime import datetime

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        MCFCRN_model = MCFCRN(len(PRE.characters), PRE.input_shape)
        MCFCRN_model.compile(optimizer=Adam(learning_rate=lr_schedule))
        #MCFCRN_model.compile(optimizer=Adadelta(learning_rate=0.0001, rho=0.9))

    MCFCRN_model.fit(train_data, epochs=200, shuffle=True)
    MCFCRN_model.save(f"MCFCRN_{round(datetime.now().timestamp())}")
    
    #%%
    MCFCRN_prediction_model = tf.keras.models.Model(
        MCFCRN_model.get_layer(name="images").input, MCFCRN_model.get_layer(name="soft_out").output
    )
    y_MCFCRN = MCFCRN_prediction_model.predict(test_data)
    y_MCFCRN_decode = PRE.model_decoder(y_MCFCRN)
    
    #%%
    for i, (f, label) in enumerate(test_data):
        print(f"{i}_actual_pred")
        print(label.numpy()[0].decode("utf-8"))
        print(y_MCFCRN_decode[i])