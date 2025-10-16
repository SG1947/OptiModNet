import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as L
from math import log2
from .layers import channel_attention, transformer_encoder, conv_block, deconv_block

def build_unetr_2d(cf):
    input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    inputs = L.Input(input_shape)
    patch_embed = L.Dense(cf["hidden_dim"])(inputs)
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)
    x = patch_embed + pos_embed

    skip_connection_index = [3, 6, 9, 12]
    skip_connections = []

    for i in range(1, cf["num_layers"]+1):
        x = transformer_encoder(x, cf)
        if i in skip_connection_index:
            skip_connections.append(x)

    z3, z6, z9, z12 = skip_connections
    z0 = L.Reshape((cf["image_size"], cf["image_size"], cf["num_channels"]))(inputs)
    shape = (cf["image_size"] // cf["patch_size"], cf["image_size"] // cf["patch_size"], cf["hidden_dim"])
    z3 = L.Reshape(shape)(z3)
    z6 = L.Reshape(shape)(z6)
    z9 = L.Reshape(shape)(z9)
    z12 = L.Reshape(shape)(z12)

    total_upscale_factor = int(log2(cf["patch_size"]))
    upscale = total_upscale_factor - 4

    if upscale >= 2:
        z3 = deconv_block(z3, z3.shape[-1], strides=2**upscale)
        z6 = deconv_block(z6, z6.shape[-1], strides=2**upscale)
        z9 = deconv_block(z9, z9.shape[-1], strides=2**upscale)
        z12 = deconv_block(z12, z12.shape[-1], strides=2**upscale)
    elif upscale < 0:
        p = 2**abs(upscale)
        z3 = L.MaxPool2D((p, p))(z3)
        z6 = L.MaxPool2D((p, p))(z6)
        z9 = L.MaxPool2D((p, p))(z9)
        z12 = L.MaxPool2D((p, p))(z12)

    ## Decoder
    def decoder_step(x, s, filters, output_name=None):
        x = deconv_block(x, filters)
        for _ in range(4 - filters.bit_length() + 4):
            s = deconv_block(s, filters)
            s = conv_block(s, filters)
        x = L.Concatenate()([x, s])
        x = channel_attention(x)
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        if output_name:
            return L.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid", name=output_name)(x), x
        return x

    out1, x = decoder_step(z12, z9, 128, "out1")
    out2, x = decoder_step(x, z6, 64, "out2")
    out3, x = decoder_step(x, z3, 32, "out3")
    final_output, _ = decoder_step(x, z0, 16, "final_output")

    out1 = L.Resizing(cf["image_size"], cf["image_size"], name="out1_resize")(out1)
    out2 = L.Resizing(cf["image_size"], cf["image_size"], name="out2_resize")(out2)
    out3 = L.Resizing(cf["image_size"], cf["image_size"], name="out3_resize")(out3)

    return Model(inputs, [out1, out2, out3, final_output], name="UNETR_2D")
