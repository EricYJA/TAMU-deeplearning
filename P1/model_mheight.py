from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# The nine (n, k, m) tuples that appear in the dataset.
SUPPORTED_COMBOS = [
    (9, 4, 2),
    (9, 4, 3),
    (9, 4, 4),
    (9, 4, 5),
    (9, 5, 2),
    (9, 5, 3),
    (9, 5, 4),
    (9, 6, 2),
    (9, 6, 3),
]


LOG_EPS = 1e-6
L2_WEIGHT_DECAY = 1e-5


def heteroscedastic_gaussian_nll_from_log(
    y_true_log: tf.Tensor,
    y_pred_params: tf.Tensor,
    min_log_var: float = -6.0,
    max_log_var: float = 4.0,
    variance_prior_alpha: float = 1e-3,
) -> tf.Tensor:
    """Gaussian negative log-likelihood with learned log-variance and variance prior penalty."""
    mean_log = y_pred_params[..., 0]
    raw_log_var = y_pred_params[..., 1]
    log_var = tf.clip_by_value(raw_log_var, min_log_var, max_log_var)
    inv_var = tf.exp(-log_var)
    loss = 0.5 * (log_var + tf.square(y_true_log - mean_log) * inv_var)
    variance_penalty = variance_prior_alpha * tf.square(tf.nn.relu(min_log_var - raw_log_var))
    return tf.reduce_mean(loss + variance_penalty)


def student_t_nll_from_log(
    y_true_log: tf.Tensor,
    y_pred_params: tf.Tensor,
    min_log_var: float = -5.0,
    max_log_var: float = 3.0,
    min_df: float = 2.0,
    max_df: float = 120.0,
    variance_prior_alpha: float = 1e-3,
) -> tf.Tensor:
    """Student-t negative log-likelihood tailored for heteroscedastic regression."""
    loc = y_pred_params[..., 0]
    raw_log_var = y_pred_params[..., 1]
    log_var = tf.clip_by_value(raw_log_var, min_log_var, max_log_var)
    scale = tf.exp(0.5 * log_var)

    raw_df = y_pred_params[..., 2]
    df = tf.nn.softplus(raw_df) + min_df
    if max_df is not None:
        df = tf.minimum(df, max_df)

        half_df_plus_one = 0.5 * (df + 1.0)
        
    half_df = 0.5 * df
    log_pi = tf.math.log(tf.constant(np.pi, dtype=y_true_log.dtype))
    log_scale = 0.5 * log_var
    scaled_sq_error = tf.square(y_true_log - loc) * tf.exp(-log_var)
    log_unnormalized = tf.math.lgamma(half_df_plus_one) - tf.math.lgamma(half_df)
    log_normalization = 0.5 * (tf.math.log(df) + log_pi) + log_scale
    log_kernel = half_df_plus_one * tf.math.log1p(scaled_sq_error / df)
    log_prob = log_unnormalized - log_normalization - log_kernel
    nll = -log_prob
    variance_penalty = variance_prior_alpha * tf.square(tf.nn.relu(min_log_var - raw_log_var))
    return tf.reduce_mean(nll + variance_penalty)


def symmetric_relative_loss(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    scale = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0 + eps
    return tf.reduce_mean(tf.square((y_pred - y_true) / scale))


def symmetric_relative_loss_from_log(y_true_log: tf.Tensor, y_pred_log: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    y_true = tf.pow(2.0, y_true_log)
    y_pred = tf.pow(2.0, y_pred_log)
    return symmetric_relative_loss(y_true, y_pred, eps=eps)


def to_log2_height(values: np.ndarray, eps: float = LOG_EPS) -> np.ndarray:
    safe = np.maximum(values, eps)
    return np.log2(safe)


def from_log2_height(values: np.ndarray) -> np.ndarray:
    return np.power(2.0, values)


def unpack_log_prediction(predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split model outputs into log-mean and log-variance arrays."""
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        return preds, np.zeros_like(preds)
    if preds.shape[-1] == 1:
        return preds[..., 0], np.zeros_like(preds[..., 0])
    return preds[..., 0], preds[..., 1]


def symmetric_relative_loss_from_log_params(
    y_true_log: tf.Tensor,
    y_pred_params: tf.Tensor,
    eps: float = 1e-7,
) -> tf.Tensor:
    """Convenience metric: symmetric relative error using predicted mean."""
    return symmetric_relative_loss_from_log(y_true_log, y_pred_params[..., 0], eps=eps)


class TransformerBlock(layers.Layer):
    """Tiny wrapper around a standard Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        key_dim = embed_dim // num_heads
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        hidden_dim = int(embed_dim * mlp_ratio)
        kernel_reg = regularizers.L2(L2_WEIGHT_DECAY)
        self.ffn_dense1 = layers.Dense(hidden_dim, activation=tf.nn.gelu, kernel_regularizer=kernel_reg)
        self.ffn_dropout = layers.Dropout(dropout)
        self.ffn_dense2 = layers.Dense(embed_dim, kernel_regularizer=kernel_reg)
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_output = self.attn(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(inputs + attn_output)

        ffn_output = self.ffn_dense1(x)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x + ffn_output)


class ColumnWiseSetEncoder(layers.Layer):
    """Permutation-equivariant encoder across the column dimension."""

    def __init__(self, column_dim: int, embed_dim: int, num_heads: int, depth: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        kernel_reg = regularizers.L2(L2_WEIGHT_DECAY)
        self.projection = layers.Dense(embed_dim, kernel_regularizer=kernel_reg)
        self.blocks = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ]
        self.dropout = layers.Dropout(dropout)
        self.pool_gate = layers.Dense(1, kernel_regularizer=kernel_reg)
        self.pool_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pool_out = layers.Dense(embed_dim, activation=tf.nn.gelu, kernel_regularizer=kernel_reg)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.projection(inputs)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.dropout(x, training=training)
        attn_logits = self.pool_gate(x)
        attn_weights = tf.nn.softmax(attn_logits, axis=1)
        attn_summary = tf.reduce_sum(attn_weights * x, axis=1)
        mean_summary = tf.reduce_mean(x, axis=1)
        max_summary = tf.reduce_max(x, axis=1)
        pooled = tf.concat([attn_summary, mean_summary, max_summary], axis=-1)
        pooled = self.pool_norm(pooled)
        return self.pool_out(pooled)


class RowWiseEncoder(layers.Layer):
    """Encoder that reasons across the row dimension."""

    def __init__(self, embed_dim: int, num_heads: int, depth: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        kernel_reg = regularizers.L2(L2_WEIGHT_DECAY)
        self.projection = layers.Dense(embed_dim, kernel_regularizer=kernel_reg)
        self.position_proj = layers.Dense(embed_dim, use_bias=False, kernel_regularizer=kernel_reg)
        self.blocks = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ]
        self.dropout = layers.Dropout(dropout)
        self.out_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.projection(inputs)
        row_count = tf.shape(inputs)[1]
        row_positions = tf.linspace(-1.0, 1.0, row_count)
        row_positions = tf.cast(row_positions, inputs.dtype)
        row_positions = row_positions[tf.newaxis, :, tf.newaxis]
        batch = tf.shape(inputs)[0]
        row_positions = tf.tile(row_positions, [batch, 1, 1])
        x = x + self.position_proj(row_positions)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.dropout(x, training=training)
        summary = tf.reduce_mean(x, axis=1)
        return self.out_norm(summary)


class MHeightRegressor(tf.keras.Model):
    """Model that consumes matrix P for a fixed (n, k, m) combo and predicts m_h."""

    def __init__(
        self,
        column_dim: int,
        embed_dim: int = 64,
        attn_depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.column_encoder = ColumnWiseSetEncoder(
            column_dim=column_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=attn_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.row_encoder = RowWiseEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=attn_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.head_norm = layers.LayerNormalization(epsilon=1e-6)
        kernel_reg = regularizers.L2(L2_WEIGHT_DECAY)
        self.head_dense1 = layers.Dense(embed_dim, activation=tf.nn.gelu, kernel_regularizer=kernel_reg)
        self.head_dropout = layers.Dropout(dropout)
        self.head_dense2 = layers.Dense(embed_dim // 2, activation=tf.nn.gelu, kernel_regularizer=kernel_reg)
        self.head_out = layers.Dense(3, kernel_regularizer=kernel_reg)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        column_tokens = tf.transpose(inputs, perm=[0, 2, 1])
        column_features = self.column_encoder(column_tokens, training=training)
        row_features = self.row_encoder(inputs, training=training)
        features = tf.concat([column_features, row_features], axis=-1)
        features = self.head_norm(features)
        x = self.head_dropout(self.head_dense1(features), training=training)
        x = self.head_dense2(x)
        x = self.head_out(x)
        return x


def build_model(k_dim: int, column_count: int) -> tf.keras.Model:
    matrix_input = tf.keras.Input(shape=(k_dim, column_count), name="matrix")
    core = MHeightRegressor(column_dim=column_count)
    outputs = core(matrix_input)
    model = tf.keras.Model(inputs=matrix_input, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(
        optimizer=optimizer,
        loss=student_t_nll_from_log,
        metrics=[symmetric_relative_loss_from_log_params],
    )
    return model
