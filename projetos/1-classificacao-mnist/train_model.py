import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

def load_mnist():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normaliza os valores dos pixels para o intervalo [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Adiciona o canal de cor: (28, 28) -> (28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return x_train, y_train, x_test, y_test


def build_model():

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Classificador
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

    return model


def main():

    x_train, y_train, _, _ = load_mnist()

    # Reserva 10% dos dados de treinamento para validar
    validation_split = 0.1

    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Interrompe o treinamento quando a val_loss para de melhorar
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("\nIniciando treinamento do modelo MNIST...\n")

    # Treina a CNN
    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=15,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )

    best_val_accuracy = max(history.history["val_accuracy"])

    print("\nTreinamento concluído.")
    print(f"Melhor acurácia de validação: {best_val_accuracy:.4f}")

    model.save("model.h5")

    print("Modelo salvo com sucesso em: model.h5")


if __name__ == "__main__":
    main()
