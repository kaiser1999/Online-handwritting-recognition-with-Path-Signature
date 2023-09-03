import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from tensorflow.keras.regularizers import L2

from MCFCRN import *

#%%
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
    
    model = tf.keras.Model(inputs=[img, labels], outputs=output, name="Transfer")
    if print_summary:
        model.summary()
    return model

#%%
from tensorflow.keras.models import load_model
from datetime import datetime

MCFCRN_model = load_model("MCFCRN")

#%%
base_model = tf.keras.models.Model(
    MCFCRN_model.get_layer(name="images").input, MCFCRN_model.get_layer(name="concat").output
)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    transfer_model = transfer_learning(base_model, len(PRE.characters), PRE.input_shape)
    transfer_model.compile(optimizer=Adam(learning_rate=lr_schedule))
    #transfer_model.compile(optimizer=Adadelta(learning_rate=0.0001, rho=0.9))

transfer_model.fit(train_data, epochs=200, shuffle=True)
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