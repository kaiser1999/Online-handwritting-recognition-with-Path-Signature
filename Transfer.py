import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from tensorflow.keras.regularizers import L2

from MCFCRN import *

#%%
def transfer_learning(total_words, input_shape, print_summary=False):
    img = tf.keras.Input(name="images", shape=input_shape, dtype="float32")
    labels = tf.keras.Input(name="label", shape=(None,), dtype="float32")
    
    x = Dense(1024, use_bias=False)(img) # Embedding layer with float inputs
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
from tensorflow.keras.models import load_model
from datetime import datetime

MCFCRN_model = load_model("MCFCRN_1330547")   # change the name if necessary

#%%
base_model = tf.keras.models.Model(
    MCFCRN_model.get_layer(name="images").input, MCFCRN_model.get_layer(name="concat").output
)

def MCFCRN_predict(features, labels):
    features = tf.expand_dims(features, axis=0)
    features = base_model(features)
    features = tf.squeeze(features, axis=0)
    return features, labels

train_MCFCRN = tf.data.Dataset.from_generator(PRE.read_path, (tf.float32, tf.float32), 
                                              args=(train_path,))

train_MCFCRN = train_MCFCRN.map(MCFCRN_predict, num_parallel_calls=tf.data.AUTOTUNE)
train_MCFCRN = train_MCFCRN.map(create_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_MCFCRN = train_MCFCRN.shuffle(buffer_size=PRE.total_entries//100).batch(1)

#%%
test_MCFCRN = tf.data.Dataset.from_generator(PRE.read_path, (tf.float32, tf.string), 
                                             args=(test_path,False))
test_MCFCRN = test_MCFCRN.map(MCFCRN_predict, num_parallel_calls=tf.data.AUTOTUNE).batch(1)
f, l = next(iter(test_MCFCRN))
input_shape = f.shape[1:]       # remove batch dimension

#%%

transfer_model = transfer_learning(len(PRE.characters), input_shape)
transfer_model.compile(optimizer=Adam(learning_rate=lr_schedule))
#transfer_model.compile(optimizer=Adadelta(learning_rate=0.0001, rho=0.9))
transfer_model.fit(train_MCFCRN, epochs=200, shuffle=True)
transfer_model.save(f"Transfer_{round(datetime.now().timestamp())}")

#%%
transfer_prediction_model = tf.keras.models.Model(
    transfer_model.get_layer(name="images").input, transfer_model.get_layer(name="soft_out").output
)

y_transfer = transfer_prediction_model.predict(test_data)
y_transfer_decode = PRE.model_decoder(y_transfer)

#%%
for i, (f, label) in enumerate(test_data):
    print(f"{i}_actual_pred")
    print(label.numpy()[0].decode("utf-8"))
    print(y_transfer_decode[i])