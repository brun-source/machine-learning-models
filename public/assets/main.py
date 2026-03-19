import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return np.load(file_path, allow_pickle=True)

def preprocess_data(data):
    data = data[:, :-1]
    return StandardScaler().fit_transform(data)

def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test, verbose=0)

def main():
    data = load_data('data.npy')
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    model = create_model()
    train_model(model, X_train, y_train, epochs=10)
    print(evaluate_model(model, X_test, y_test))

if __name__ == '__main__':
    main()