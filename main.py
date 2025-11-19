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
    def __init__(self, items, item_size, extra_size, activation='relu'):
        assert items > 0
        assert item_size > 0
        assert extra_size >= 0
        self.items = items
        self.item_size = item_size
        self.extra_size = extra_size
        self.activation_name = activation
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]
        self.weights = None
        self._expand_op = None
        # Для backpropagation
        self.last_input = None
        self.last_masked = None
        self.last_activated = None
        self.weights_grad = None

    def build(self):
        # He initialization для ReLU, Xavier для других
        if self.activation_name == 'relu':
            std = np.sqrt(2 / self.items)
        else:
            std = np.sqrt(1 / self.items)

        self.weights = np.random.randn(self.items) * std
        self.weights = self.weights.astype(np.float32)
        self.weights_grad = np.zeros_like(self.weights)

        # Создание матрицы расширения
        self._expand_op = np.zeros((self.items, self.items * self.item_size), dtype=np.float32)
        for i in range(self.items):
            start_idx = i * self.item_size
            end_idx = (i + 1) * self.item_size
            self._expand_op[i, start_idx:end_idx] = 1.0

    def call(self, inputs):
        self.last_input = np.array(inputs, dtype=np.float32)

        op_mask_part = self.last_input[:self.items * self.item_size]
        self.ext_part = self.last_input[self.items * self.item_size:] if self.extra_size > 0 else None

        # Применение весов и расширение
        extended_weights = np.matmul(self.weights, self._expand_op)

        # Сохраняем для backpropagation
        self.extended_weights = extended_weights

        # Элементное умножение
        self.last_masked = op_mask_part * extended_weights

        # Применение активации
        self.last_activated = self.activation(self.last_masked)

        if self.extra_size > 0:
            return np.concatenate((self.last_activated, self.ext_part))
        else:
            return self.last_activated

    def backward(self, output_grad):
        """Обратное распространение для этого слоя"""
        if self.extra_size > 0:
            # Разделяем градиент для основной части и дополнительной
            grad_main = output_grad[:self.items * self.item_size]
            grad_ext = output_grad[self.items * self.item_size:]
        else:
            grad_main = output_grad
            grad_ext = None

        # Градиент через активацию
        grad_through_activation = grad_main * self.activation_derivative(self.last_masked)

        # Градиент по расширенным весам (для каждого элемента)
        grad_wrt_extended_weights = grad_through_activation * self.last_input[:self.items * self.item_size]

        # Градиент по весам (агрегация по соответствующим позициям)
        grad_wrt_weights = np.zeros(self.items, dtype=np.float32)
        for i in range(self.items):
            start_idx = i * self.item_size
            end_idx = (i + 1) * self.item_size
            grad_wrt_weights[i] = np.sum(grad_wrt_extended_weights[start_idx:end_idx])

        self.weights_grad = grad_wrt_weights

        # Градиент по входу
        input_grad = np.zeros_like(self.last_input)
        input_grad_main = grad_through_activation * self.extended_weights

        input_grad[:self.items * self.item_size] = input_grad_main
        if self.extra_size > 0 and grad_ext is not None:
            input_grad[self.items * self.item_size:] = grad_ext

        return input_grad

    def apply_gradients(self, learning_rate, optimizer_state=None):
        """Применение градиентов к весам"""
        self.weights -= learning_rate * self.weights_grad
        # Сброс градиентов
        self.weights_grad = np.zeros_like(self.weights_grad)

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
            'weights': [layer.save_weights().tolist() for layer in self.layers]
        }
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
            if (saved_config['items'] != current_config['items'] or
                    saved_config['item_size'] != current_config['item_size'] or
                    saved_config['extra_size'] != current_config['extra_size']):
                print(f"Error: Layer {i} configuration mismatch. Cannot load weights.")
                return

            # Загрузка весов
            layer.load_weights(np.array(config['weights'][i], dtype=np.float32))
        print(f"\nWeights loaded from {filename}.json")


class RLNetwork(Network):
    """RL-специализированная нейросеть"""

    def __init__(self, layers, action_size=5):
        super().__init__(layers)
        self.action_size = action_size
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        self.replay_buffer = ReplayBuffer(max_size=10000)
        self.gamma = 0.99  # discount factor
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
            target_value = rewards[i]
            if not dones[i]:
                target_value += self.gamma * np.max(next_q)

            target_q[i][actions[i]] = target_value

        # Обратное распространение для каждого примера в батче
        total_grad = [np.zeros_like(layer.weights) for layer in self.layers]

        for i in range(batch_size):
            # Forward pass для сохранения промежуточных значений
            _ = self.call(states[i])

            # Вычисление градиента потерь
            output_grad = 2 * (current_q[i] - target_q[i]) / batch_size

            # Backward pass
            grad = output_grad
            for layer in reversed(self.layers):
                grad = layer.backward(grad)

            # Накопление градиентов
            for j, layer in enumerate(self.layers):
                total_grad[j] += layer.weights_grad

        # Применение градиентов с Adam оптимизацией
        for i, layer in enumerate(self.layers):
            layer_id = f"layer_{i}"
            layer.weights = self.optimizer.update(layer_id, layer.weights, total_grad[i])

        # Вычисление потерь для отладки
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
        """Основной цикл обучения RL-агента"""
        epsilon = 1.0  # начальное значение epsilon
        epsilon_min = 0.01
        epsilon_decay = 0.995

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

            loss = self.replay()

            # Уменьшение epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if episode % 10 == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Loss: {loss:.4f}, Epsilon: {epsilon:.4f}")

            # Сохранение модели каждые 100 эпизодов
            if episode % 100 == 0 and episode > 0:
                self.save(f"rl_model_episode_{episode}")

        self.save("trained_rl_model")
        print("Training completed!")


def create_rl_network():
    """Создает RL-нейросеть"""
    layers = [
        Layer(items=5, item_size=1, extra_size=0, activation='sigmoid'),
        Layer(items=5, item_size=1, extra_size=0, activation='sigmoid'),
        Layer(items=5, item_size=1, extra_size=0, activation='sigmoid')  # линейный выход для Q-значений
    ]
    return RLNetwork(layers, action_size=5)


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