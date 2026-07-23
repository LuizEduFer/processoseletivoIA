# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Luiz Eduardo Fernandes Cruz

### 1️⃣ Resumo da Arquitetura do Modelo

Foi implementada uma rede neural convolucional para classificar os dígitos manuscritos do dataset MNIST. O modelo possui três blocos convolucionais, cada um sendo formado por uma camada Conv2D, seguida de BatchNormalization e MaxPooling2D. Após os blocos convolucionais, foi utilizada uma camada Flatten, seguida por uma camada totalmente conectada (Dense) com 128 neurônios e ativação ReLU. Antes da camada de saída, foi aplicado Dropout com taxa de 0,5 para ajudar na regularização do modelo. A camada de saída possui 10 neurônios com ativação softmax, correspondendo às classes dos dígitos de 0 a 9. Os dados de treinamento foram normalizados para o intervalo [0, 1] e foi utilizado um validation_split de 10% para separar os dados de validação. O treinamento foi configurado para no máximo 15 épocas e utilizou EarlyStopping monitorando a perda de validação (val_loss), com a restauração dos melhores pesos encontrados durante o treinamento.

### 2️⃣ Bibliotecas Utilizadas

Python 3.10
TensorFlow 2.21.0
Keras 3.12.3, integrado ao TensorFlow
NumPy 2.2.6

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a técnica de Dynamic Range Quantization durante a conversão do modelo Keras para o TensorFlow Lite. No arquivo optimize_model.py, o modelo treinado em model.h5 é carregado utilizando TensorFlow e convertido por meio de tf.lite.TFLiteConverter. A otimização é habilitada com tf.lite.Optimize.DEFAULT, permitindo que o modelo seja convertido para um formato mais adequado para execução em dispositivos de Edge AI, reduzindo o tamanho e mantendo uma boa capacidade de inferência.

### 4️⃣ Resultados Obtidos

A melhor acurácia de validação obtida durante o treinamento foi de 99,00%. Os tamanhos dos modelos gerados foram:

model.h5: aproximadamente 2,84 MiB
model.tflite: aproximadamente 250,41 KiB

A conversão para TensorFlow Lite reduziu bastante o tamanho do modelo, tornando ele mais adequado para aplicações de Edge AI. Na validação automática, o modelo model.h5 apresentou 100,00% de acurácia nas 300 amostras de teste utilizadas pelo validador, enquanto o modelo model.tflite também apresentou 100,00% de acurácia nas mesmas 300 amostras. Na etapa de inferência de exemplo, foram testadas cinco amostras do conjunto de teste utilizando especificamente o modelo otimizado model.tflite por meio de tf.lite.Interpreter. Todas as cinco amostras foram classificadas corretamente.

### 5️⃣ Comentários Adicionais (Opcional)

Uma decisão importante foi utilizar uma arquitetura CNN simples, conforme as restrições do projeto, evitando arquiteturas muito profundas. O uso de EarlyStopping permitiu interromper o treinamento quando a perda de validação parou de apresentar melhora, além de restaurar os melhores pesos encontrados.

A etapa de otimização apresentou uma redução significativa no tamanho do modelo, mostrando a importância da conversão dos formatos para dispositivos com menos recursos computacionais.

### 6️⃣ Exemplo de Inferência

A execução do arquivo run_inference.py utilizando o modelo otimizado model.tflite apresentou os seguintes resultados:

Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Nas cinco amostras selecionadas, o modelo apresentou acerto em todas as previsões, com resultado de 5 acertos em 5 amostras. Nesse conjunto específico de inferência, não foram observados casos de erro.
