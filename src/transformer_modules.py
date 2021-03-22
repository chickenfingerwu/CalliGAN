import numpy as np
import tensorflow.compat.v1 as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(x, max_seq_len):
    shape = x.get_shape().as_list()
    d_model = shape[-1]
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def attention(q, k, v, mask=None, dropout_rate=None):
    """
    q: batch_size x head x seq_length x d_model
    k: batch_size x head x seq_length x d_model
    v: batch_size x head x seq_length x d_model
    mask: batch_size x 1 x 1 x seq_length
    output: batch_size x head x seq_length x d_model
    """

    # attention score được tính bằng cách nhân q với k
    d_k = q.size(-1)
    scores = tf.linalg.matmul(q, k.transpose(-2, -1)) / tf.math.sqrt(d_k))

    if mask is not None:
        mask = tf.expand_dims(mask,1)
        scores = tf.where(mask == 0, -1e9, mask)
    # xong rồi thì chuẩn hóa bằng softmax
    scores = tf.nn.softmax(scores, dim=-1)

    if dropout_rate is not None:
        scores = tf.nn.dropout(scores, dropout_rate)

    output = tf.linalg.matmul.matmul(scores, v)
    return output, scores

def multihead_attention(queries, keys, values, key_masks, num_heads=8,
                        dropout_rate=0.3,training=False,causality=True):
    shape = queries.get_shape().as_list()
    d_model = shape[-1]
    assert d_model % num_heads == 0
    d_k = d_model // num_heads
    bs = shape[0]

    q_linear = tf.layers.dense(queries, trainable=training)
    q_linear = tf.reshape(q_linear, tf.constant([bs, -1, num_heads, d_k]))
    k_linear = tf.layers.dense(keys, trainable=training)
    k_linear = tf.reshape(k_linear, tf.constant([bs, -1, num_heads, d_k]))
    v_linear = tf.layers.dense(values, trainable=training)
    v_linear = tf.reshape(v_linear, tf.constant([bs, -1, num_heads, d_k]))

    q = tf.transpose(q_linear)
    v = tf.transpose(v_linear)
    k = tf.transpose(k_linear)

    scores, attn = attention(q, k, v, key_masks, dropout_rate)

    concat = tf.transpose(tf.reshape(scores, tf.constant([bs, -1, num_heads, d_k])))
    return tf.layers.dense(concat)