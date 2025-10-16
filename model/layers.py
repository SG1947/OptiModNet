import tensorflow.keras.layers as L
import tensorflow as tf

def channel_attention(input_tensor, reduction_ratio=16):
    channel_avg = L.GlobalAveragePooling2D()(input_tensor)
    channel_avg = L.Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu')(channel_avg)
    channel_avg = L.Dense(input_tensor.shape[-1], activation='sigmoid')(channel_avg)
    channel_avg = L.Reshape((1, 1, input_tensor.shape[-1]))(channel_avg)
    return input_tensor * channel_avg

def mlp(x, cf):
    x = L.Dense(cf["mlp_dim"], activation="gelu")(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    x = L.Dense(cf["hidden_dim"])(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    return x

def transformer_encoder(x, cf):
    skip = x
    x = L.LayerNormalization()(x)
    x = L.GroupQueryAttention(head_dim=cf["head_dim"], num_query_heads=cf["num_query_heads"], num_key_value_heads=cf["num_key_value_heads"])(x, x)
    x = L.Add()([x, skip])
    skip = x
    x = L.LayerNormalization()(x)
    x = mlp(x, cf)
    x = L.Add()([x, skip])
    return x

def conv_block(x, num_filters):
    x = L.Conv2D(num_filters, [3, 3], padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x

def deconv_block(x, num_filters, strides=2):
    x = L.Conv2DTranspose(num_filters, kernel_size=2, padding="same", strides=strides)(x)
    return x
