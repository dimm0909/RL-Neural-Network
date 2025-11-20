import numpy as np
import json
import os
import random
from collections import deque


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


ACTIVATION_FUNCTIONS = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'none': lambda x: x
}

ACTIVATION_DERIVATIVES = {
    'relu': relu_derivative,
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'none': lambda x: np.ones_like(x)
}


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        assert input_size > 0
        assert output_size > 0
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]
        self.weights = None
        self.bias = None
        self.last_input = None
        self.last_weighted_sum = None
        self.last_activated = None
        self.weights_grad = None
        self.bias_grad = None

    def build(self):
        # Xavier/Glorot initialization
        std = np.sqrt(2 / (self.input_size + self.output_size))

        # Веса: матрица [output_size, input_size]
        self.weights = np.random.randn(self.output_size, self.input_size) * std
        self.weights = self.weights.astype(np.float32)

        # Смещения
        self.bias = np.zeros(self.output_size, dtype=np.float32)

        # Градиенты
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

    def call(self, inputs):
        self.last_input = np.array(inputs, dtype=np.float32).reshape(-1)

        # Проверка размера входа
        if len(self.last_input) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {len(self.last_input)}")

        # Линейное преобразование: output = weights @ input + bias
        self.last_weighted_sum = np.dot(self.weights, self.last_input) + self.bias

        # Применение активации
        self.last_activated = self.activation(self.last_weighted_sum)

        return self.last_activated.copy()

    def backward(self, output_grad):
        """Обратное распространение для этого слоя"""
        # Убедимся, что градиент имеет правильную размерность
        if output_grad.ndim == 1 and output_grad.shape[0] != self.output_size:
            # Если градиент не соответствует размеру выхода, преобразуем его
            output_grad = output_grad.reshape(-1)
            if output_grad.shape[0] != self.output_size:
                raise ValueError(f"Gradient size mismatch: expected {self.output_size}, got {output_grad.shape[0]}")

        # Градиент через активацию
        grad_through_activation = output_grad * self.activation_derivative(self.last_weighted_sum)

        # Градиент по весам
        self.weights_grad = np.outer(grad_through_activation, self.last_input)

        # Градиент по смещениям
        self.bias_grad = grad_through_activation.copy()

        # Градиент по входу
        input_grad = np.dot(self.weights.T, grad_through_activation)

        return input_grad

    def apply_gradients(self, optimizer, layer_id):
        """Применение градиентов к весам и смещениям"""
        # Обновление весов
        self.weights = optimizer.update(f"{layer_id}_weights", self.weights, self.weights_grad)
        # Обновление смещений
        self.bias = optimizer.update(f"{layer_id}_bias", self.bias, self.bias_grad)

        # Сброс градиентов
        self.weights_grad = np.zeros_like(self.weights_grad)
        self.bias_grad = np.zeros_like(self.bias_grad)

    def get_config(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_name
        }

    def save_weights(self):
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }

    def load_weights(self, weights_data):
        self.weights = weights_data['weights'].astype(np.float32)
        self.bias = weights_data['bias'].astype(np.float32)

class Optimizer:
    """Базовый класс оптимизатора"""

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, layer, grad):
        raise NotImplementedError


class AdamOptimizer(Optimizer):
    """Adam оптимизатор"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0  # timestep

    def update(self, layer_id, weights, grad):
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(weights)
            self.v[layer_id] = np.zeros_like(weights)

        self.t += 1
        self.m[layer_id] = self.beta1 * self.m[layer_id] + (1 - self.beta1) * grad
        self.v[layer_id] = self.beta2 * self.v[layer_id] + (1 - self.beta2) * grad ** 2

        m_hat = self.m[layer_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer_id] / (1 - self.beta2 ** self.t)

        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return weights - update


class ReplayBuffer:
    """Буфер воспроизведения для RL"""

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Добавление опыта в буфер"""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Сэмплирование батча из буфера"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


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
            'weights': []
        }

        # Сохраняем веса в формате, совместимом с JSON
        for layer in self.layers:
            weights_data = layer.save_weights()
            config['weights'].append({
                'weights': weights_data['weights'].tolist(),
                'bias': weights_data['bias'].tolist()
            })

        with open(f"{filename}.json", 'w') as f:
            json.dump(config, f)
        print(f"\nModel saved to {filename}.json")

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
            if (saved_config['input_size'] != current_config['input_size'] or
                    saved_config['output_size'] != current_config['output_size']):
                print(f"Error: Layer {i} configuration mismatch. Cannot load weights.")
                return

            # Загрузка весов
            weights_data = {
                'weights': np.array(config['weights'][i]['weights'], dtype=np.float32),
                'bias': np.array(config['weights'][i]['bias'], dtype=np.float32)
            }
            layer.load_weights(weights_data)
        print(f"\nWeights loaded from {filename}.json")


class RLNetwork(Network):
    """RL-специализированная нейросеть"""
    def __init__(self, layers, action_size):
        super().__init__(layers)
        self.action_size = action_size
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        self.replay_buffer = ReplayBuffer(max_size=10000)
        self.gamma = 0.99
        self.batch_size = 32

    def predict_q_values(self, state):
        """Предсказание Q-значений для состояния"""
        return self.call(state)

    def act(self, state, epsilon=0.1):
        """Выбор действия с использованием epsilon-greedy стратегии"""
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)

        q_values = self.predict_q_values(state)
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        """Сохранение опыта в буфер"""
        self.replay_buffer.add((state, action, reward, next_state, done))

    def compute_loss(self, q_pred, target_q):
        """Вычисление MSE потерь"""
        return np.mean((q_pred - target_q) ** 2)

    def train_on_batch(self, states, actions, rewards, next_states, dones):
        """Обучение на одном батче"""
        batch_size = len(states)
        current_q = np.zeros((batch_size, self.action_size))
        target_q = np.zeros((batch_size, self.action_size))

        # Получение текущих Q-значений
        for i in range(batch_size):
            current_q[i] = self.predict_q_values(states[i])
            target_q[i] = current_q[i].copy()

            # Вычисление целевых Q-значений
            next_q = self.predict_q_values(next_states[i])
            for action in range(self.action_size):
                target_value = rewards[i]
                if not dones[i]:
                    # Используем Double DQN подход для стабильности
                    best_next_action = np.argmax(next_q)
                    target_value += self.gamma * next_q[best_next_action]
                target_q[i][action] = target_value

            # Обновляем только Q-значение для выбранного действия
            target_q[i][actions[i]] = target_value

        # Обратное распространение для каждого примера в батче
        total_weights_grad = [np.zeros_like(layer.weights) for layer in self.layers]
        total_bias_grad = [np.zeros_like(layer.bias) for layer in self.layers]

        for i in range(batch_size):
            # Forward pass для сохранения промежуточных значений
            _ = self.call(states[i])

            # Вычисление градиента потерь
            output_grad = 2 * (current_q[i] - target_q[i]) / batch_size

            # Backward pass
            grad = output_grad.copy()  # Создаем копию для безопасности
            for layer in reversed(self.layers):
                grad = layer.backward(grad)

            # Накопление градиентов
            for j, layer in enumerate(self.layers):
                total_weights_grad[j] += layer.weights_grad
                total_bias_grad[j] += layer.bias_grad

        # Применение градиентов
        for i, layer in enumerate(self.layers):
            layer_id = f"layer_{i}"
            layer.weights = self.optimizer.update(f"{layer_id}_weights", layer.weights, total_weights_grad[i])
            layer.bias = self.optimizer.update(f"{layer_id}_bias", layer.bias, total_bias_grad[i])

        loss = self.compute_loss(current_q, target_q)
        return loss

    def replay(self):
        """Обучение на опыте из буфера"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        minibatch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        loss = self.train_on_batch(states, actions, rewards, next_states, dones)
        return loss

    def train(self, env, episodes=1000, max_steps=200):
        """Основной цикл обучения RL-агента с целевой сетью"""
        # Создание целевой сети
        self.update_target_network()
        target_update_freq = 100  # Частота обновления целевой сети

        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        total_steps = 0

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                total_steps += 1

                # Обучение на каждом шаге, если в буфере достаточно данных
                if len(self.replay_buffer) > self.batch_size:
                    self.replay()

                if done:
                    break

            # Уменьшение epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Обновление целевой сети
            if total_steps % target_update_freq == 0:
                self.update_target_network()
                print(f"Updated target network at step {total_steps}")

            if episode % 10 == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

            # Сохранение модели каждые 100 эпизодов
            if episode % 100 == 0 and episode > 0:
                self.save(f"rl_model_episode_{episode}")

        self.save("trained_rl_model")
        print("Training completed!")

    def update_target_network(self):
        """Создание и обновление целевой сети для стабильного обучения"""
        target_layers = []
        for layer in self.layers:
            target_layer = Layer(layer.input_size, layer.output_size, layer.activation_name)
            target_layer.build()
            target_layer.weights = layer.weights.copy()
            target_layer.bias = layer.bias.copy()
            target_layers.append(target_layer)

        self.target_network = Network(target_layers)
        return self.target_network

    def compute_target_q(self, next_states, rewards, dones):
        """Вычисление целевых Q-значений с использованием целевой сети"""
        batch_size = len(next_states)
        target_q = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            next_q = self.target_network.call(next_states[i])
            best_next_action = np.argmax(next_q)

            for action in range(self.action_size):
                target_value = rewards[i]
                if not dones[i]:
                    target_value += self.gamma * next_q[best_next_action]
                target_q[i][action] = target_value

        return target_q


def create_rl_network():
    """Создает RL-нейросеть с гибкой архитектурой"""
    layers = [
        Layer(input_size=5, output_size=16, activation='relu'),
        Layer(input_size=16, output_size=8, activation='relu'),
        Layer(input_size=8, output_size=1, activation='none')
    ]
    return RLNetwork(layers, action_size=1)


def main():
    # Создание RL-сети
    net = create_rl_network()
    net.build()

    # Загрузка весов если они существуют
    net.load("rl_model")

    print("RL Neural Network initialized!")

    while True:
        print("\n1. Run single prediction")
        print("2. Start training (requires environment)")
        print("3. Load from rl_model")
        print("4. Load from trained_rl_model")
        print("5. Exit\n")

        choice = int(input("Select option (1-5): "))

        if choice == 1:
            print("\nEnter 5 numbers separated by spaces for state:")
            try:
                user_input = list(map(float, input().split()))
                if len(user_input) != 5:
                    print("Error: Exactly 5 numbers required.")
                    return
            except ValueError:
                print("Error: Invalid input. Please enter numbers only.")
                return

            # Предсказание Q-значений
            q_values = net.predict_q_values(user_input)
            print("\nQ-values for each action:", q_values)
            best_action = np.argmax(q_values)
            print(f"Best action: {best_action} with Q-value: {q_values[best_action]:.4f}")

            # Сохранение модели
            net.save("rl_model")

        elif choice == 2:
            print("\nTraining mode selected. Note: This requires a compatible environment.")
            print("For demonstration, we'll use a simple mock environment.")

            class MockEnv:
                """Простая mock-среда для демонстрации"""

                def __init__(self):
                    self.state = np.random.rand(5)
                    self.steps = 0

                def reset(self):
                    self.state = np.random.rand(5)
                    self.steps = 0
                    return self.state

                def step(self, action):
                    self.steps += 1
                    # Простая логика вознаграждения
                    reward = float(np.sin(self.steps * 0.1) + 0.1 * action)
                    self.state = np.random.rand(5)  # новое случайное состояние
                    done = self.steps >= 50
                    return self.state, reward, done, {}

            env = MockEnv()
            print("\nStarting training with mock environment...")
            print("This is a demonstration. For real RL tasks, use proper environments like Gym.")

            net.train(env, episodes=100, max_steps=50)

        elif choice == 3:
            # Загрузка весов если они существуют
            net.load("rl_model")

        elif choice == 4:
            net.load("trained_rl_model")

        elif choice == 5:
            print("\nExiting...")
            return

        else:
            print("Invalid choice!")


if __name__ == '__main__':
    main()