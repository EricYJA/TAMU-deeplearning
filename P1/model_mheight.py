import os
import pickle
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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


def symmetric_relative_loss(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    scale = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0 + eps
    return tf.reduce_mean(tf.square((y_pred - y_true) / scale))


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
        self.ffn_dense1 = layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.ffn_dropout = layers.Dropout(dropout)
        self.ffn_dense2 = layers.Dense(embed_dim)
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
        self.projection = layers.Dense(embed_dim)
        self.blocks = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ]
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.projection(inputs)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.dropout(x, training=training)
        return tf.reduce_mean(x, axis=1)


class MHeightRegressor(tf.keras.Model):
    """Model that consumes matrix P for a fixed (n, k, m) combo and predicts m_h."""

    def __init__(
        self,
        column_dim: int,
        embed_dim: int = 128,
        attn_depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ColumnWiseSetEncoder(
            column_dim=column_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=attn_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.head_dense1 = layers.Dense(embed_dim, activation=tf.nn.gelu)
        self.head_dropout = layers.Dropout(dropout)
        self.head_dense2 = layers.Dense(embed_dim // 2, activation=tf.nn.gelu)
        self.head_out = layers.Dense(1)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        column_tokens = tf.transpose(inputs, perm=[0, 2, 1])
        set_features = self.encoder(column_tokens, training=training)
        x = self.head_dropout(self.head_dense1(set_features), training=training)
        x = self.head_dense2(x)
        x = self.head_out(x)
        return tf.squeeze(x, axis=-1)


def load_dataset(
    n_k_m_path: str,
    m_height_path: str,
) -> Tuple[Sequence[Tuple[int, int, int, np.ndarray]], Sequence[float]]:
    with open(n_k_m_path, "rb") as f:
        feature_samples = pickle.load(f)
    with open(m_height_path, "rb") as f:
        height_samples = pickle.load(f)
    return feature_samples, height_samples


def _collect_for_combo(
    combo: Tuple[int, int, int],
    samples: Sequence[Tuple[int, int, int, np.ndarray]],
    targets: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    matrices = []
    values = []
    for sample, target in zip(samples, targets):
        if tuple(sample[:3]) == combo:
            matrices.append(np.asarray(sample[3], dtype=np.float32))
            values.append(np.asarray(target, dtype=np.float32))
    if not matrices:
        raise ValueError(f"No samples available for combo {combo}.")
    return np.stack(matrices), np.stack(values)


def select_combo_data(
    combo: Tuple[int, int, int],
    samples: Sequence[Tuple[int, int, int, np.ndarray]],
    targets: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if combo == (9, 4, 2):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 4, 3):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 4, 4):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 4, 5):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 5, 2):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 5, 3):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 5, 4):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 6, 2):
        return _collect_for_combo(combo, samples, targets)
    elif combo == (9, 6, 3):
        return _collect_for_combo(combo, samples, targets)
    else:
        raise ValueError(f"Unsupported combo {combo}.")


def split_dataset(
    matrices: np.ndarray,
    targets: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    total = matrices.shape[0]
    indices = np.arange(total)
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    matrices = matrices[indices]
    targets = targets[indices]

    train_end = max(1, int(total * 0.85))
    val_end = max(train_end + 1, int(total * 0.90))
    val_end = min(val_end, total - 1) if total > 2 else total - 1
    train = (matrices[:train_end], targets[:train_end])
    val = (matrices[train_end:val_end], targets[train_end:val_end])
    test = (matrices[val_end:], targets[val_end:])

    if val[0].shape[0] == 0:
        val = train
    if test[0].shape[0] == 0:
        test = val
    return train, val, test


def build_model(k_dim: int, column_count: int) -> tf.keras.Model:
    matrix_input = tf.keras.Input(shape=(k_dim, column_count), name="matrix")
    core = MHeightRegressor(column_dim=k_dim)
    outputs = core(matrix_input)
    model = tf.keras.Model(inputs=matrix_input, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=optimizer, loss=symmetric_relative_loss)
    return model


def train_model_for_combo(
    combo: Tuple[int, int, int],
    samples: Sequence[Tuple[int, int, int, np.ndarray]],
    targets: Sequence[float],
    epochs: int = 200,
    batch_size: int = 512,
    save_dir: str = "trained_models",
) -> Dict[str, float]:
    matrices, values = select_combo_data(combo, samples, targets)
    train, val, test = split_dataset(matrices, values)

    model = build_model(k_dim=matrices.shape[1], column_count=matrices.shape[2])
    history = model.fit(
        train[0],
        train[1],
        validation_data=(val[0], val[1]),
        batch_size=min(batch_size, train[0].shape[0]),
        epochs=epochs,
        verbose=2,
    )

    train_loss = float(history.history["loss"][-1])
    val_loss = float(history.history["val_loss"][-1]) if "val_loss" in history.history else float("nan")
    test_loss = float(model.evaluate(test[0], test[1], verbose=0))

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{combo[0]}_{combo[1]}_{combo[2]}_model")
    model.save(model_path, include_optimizer=False)

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "model_path": model_path,
    }


def train_all_combos(
    samples: Sequence[Tuple[int, int, int, np.ndarray]],
    targets: Sequence[float],
    combos: Iterable[Tuple[int, int, int]] = SUPPORTED_COMBOS,
) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    results: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for combo in combos:
        print(f"Training combo {combo}...")
        results[combo] = train_model_for_combo(combo, samples, targets)
    return results


def main() -> None:
    n_k_m_path = "../data/project/DS-15-samples_n_k_m_P"
    m_height_path = "../data/project/DS-15-samples_mHeights"
    samples, targets = load_dataset(n_k_m_path, m_height_path)
    metrics = train_all_combos(samples, targets)
    for combo, stats in metrics.items():
        print(f"{combo}: {stats}")


if __name__ == "__main__":
    main()
