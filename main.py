import numpy as np
import json
import os


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


def tanh(x):
    return np.tanh(x)


ACTIVATION_FUNCTIONS = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'none': lambda x: x
}


class Layer:
    def __init__(self, items, item_size, extra_size, activation='relu'):
        assert items > 0
        assert item_size > 0
        assert extra_size >= 0
        self.items = items
        self.item_size = item_size
        self.extra_size = extra_size
        self.activation_name = activation
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.weights = None  # Learnable weights
        self._expand_op = None

    def build(self):
        # Инициализация весов с использованием метода He initialization
        self.weights = np.random.randn(self.items) * np.sqrt(2 / self.items)
        self.weights = self.weights.astype(np.float32)

        # Создание матрицы расширения
        self._expand_op = np.zeros((self.items, self.items * self.item_size), dtype=np.float32)
        for i in range(self.items):
            start_idx = i * self.item_size
            end_idx = (i + 1) * self.item_size
            self._expand_op[i, start_idx:end_idx] = 1.0

    def call(self, inputs):
        op_mask_part = inputs[:self.items * self.item_size]
        ext_part = inputs[self.items * self.item_size:] if self.extra_size > 0 else None

        # Применение весов и расширение
        extended_weights = np.matmul(self.weights, self._expand_op)

        # Элементное умножение и применение активации
        masked = op_mask_part * extended_weights
        activated = self.activation(masked)

        return np.concatenate((activated, ext_part)) if self.extra_size > 0 else activated

    def get_config(self):
        return {
            'items': self.items,
            'item_size': self.item_size,
            'extra_size': self.extra_size,
            'activation': self.activation_name
        }

    def save_weights(self):
        return self.weights.copy()

    def load_weights(self, weights):
        self.weights = weights.astype(np.float32)


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.input_layer = layers[0]
        self.output_layer = layers[-1]

    def build(self):
        for layer in self.layers:
            layer.build()

    def call(self, input_data):
        output = np.array(input_data, dtype=np.float32)
        for i, layer in enumerate(self.layers):
            output = layer.call(output)
        return output

    def save(self, filename):
        config = {
            'layers': [layer.get_config() for layer in self.layers],
            'weights': [layer.save_weights().tolist() for layer in self.layers]
        }
        with open(f"{filename}.json", 'w') as f:
            json.dump(config, f)
        print(f"Model saved to {filename}.json")

    def load(self, filename):
        if not os.path.exists(f"{filename}.json"):
            print(f"Warning: File {filename}.json not found. Skipping load.")
            return

        with open(f"{filename}.json", 'r') as f:
            config = json.load(f)

        # Проверка совместимости структуры
        if len(config['layers']) != len(self.layers):
            print("Error: Layer count mismatch. Cannot load weights.")
            return

        for i, layer in enumerate(self.layers):
            saved_config = config['layers'][i]
            current_config = layer.get_config()

            # Проверка совместимости параметров слоя
            if (saved_config['items'] != current_config['items'] or
                    saved_config['item_size'] != current_config['item_size'] or
                    saved_config['extra_size'] != current_config['extra_size']):
                print(f"Error: Layer {i} configuration mismatch. Cannot load weights.")
                return

            # Загрузка весов
            layer.load_weights(np.array(config['weights'][i], dtype=np.float32))
        print(f"Weights loaded from {filename}.json")


def create_network():
    """Создает предопределенную архитектуру сети"""
    layers = [
        Layer(items=5, item_size=1, extra_size=0, activation='sigmoid'),
        Layer(items=5, item_size=1, extra_size=0, activation='sigmoid'),
        Layer(items=5, item_size=1, extra_size=0, activation='sigmoid')
    ]
    return Network(layers)


def main():
    # Создание и инициализация сети
    net = create_network()
    net.build()

    # Загрузка весов если существует сохраненная модель
    net.load("rl_model")

    print("Enter 5 numbers separated by spaces:")
    try:
        user_input = list(map(float, input().split()))
        if len(user_input) != 5:
            print("Error: Exactly 5 numbers required.")
            return
    except ValueError:
        print("Error: Invalid input. Please enter numbers only.")
        return

    # Прямой проход через сеть
    result = net.call(user_input)
    print("Network output:", result)

    # Сохранение весов после использования
    net.save("rl_model")


if __name__ == '__main__':
    main()