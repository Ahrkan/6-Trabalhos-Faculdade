# Importação de bibliotecas necessárias
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Carregar o arquivo CSV com os dados de AIDS
data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/Aids%20Data.csv')

# Explorar os dados
print(data.head())  # Mostrar as primeiras linhas do dataset

# Pré-processamento dos dados
# Verificar valores nulos
print(data.isnull().sum())

# Remover ou preencher valores nulos, se existirem
data = data.fillna(0)

# Converter colunas categóricas em variáveis binárias usando get_dummies, se necessário
data_enum = pd.get_dummies(data)

# Separar os dados em recursos (features) e rótulos (label)
# Aqui assumimos que a coluna de rótulo (diagnóstico de AIDS) é "Diagnosis"
# Verifique o nome correto da coluna no dataset
features = data_enum.drop(columns=['Diagnosis'])  # Remova a coluna de diagnóstico
labels = data_enum['Diagnosis']  # Defina o rótulo (alvo) como a coluna de diagnóstico

# Normalizar os dados para a rede neural
scaler = MinMaxScaler(feature_range=(-1, 1))
features = scaler.fit_transform(features)

# Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Construção do modelo de rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation=tf.nn.relu),  # Primeira camada densa com 64 neurônios
    tf.keras.layers.Dense(32, activation=tf.nn.relu),  # Segunda camada com 32 neurônios
    tf.keras.layers.Dense(16, activation=tf.nn.relu),  # Terceira camada com 16 neurônios
    tf.keras.layers.Dense(1, activation='sigmoid')     # Saída com 1 neurônio e função de ativação sigmoide (para classificação binária)
])

# Compilação do modelo
model.compile(optimizer='adam', 
              loss='binary_crossentropy',  # Perda adequada para problemas de classificação binária
              metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Avaliação do modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Visualização dos resultados
plt.figure(figsize=(12, 6))

# Histórico de acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia Teste')
plt.title('Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Histórico de perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda Treinamento')
plt.plot(history.history['val_loss'], label='Perda Teste')
plt.title('Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()
