import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

model = tf.keras.models.load_model("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model.tflite", "wb") as file:
    file.write(tflite_model)

# Exibe informações sobre os arquivos gerados
h5_size = os.path.getsize("model.h5")
tflite_size = os.path.getsize("model.tflite")

print("Conversão concluída com sucesso.")
print(f"Modelo original: {h5_size / (1024 * 1024):.2f} MB")
print(f"Modelo otimizado: {tflite_size / (1024 * 1024):.2f} MB")
print("Modelo salvo em: model.tflite")
