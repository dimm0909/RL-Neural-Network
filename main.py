import numpy as np


def RandomOps(count):
    ops = np.random.randint(0, 2, (count,), dtype="int")
    ops = ops.astype(dtype=np.float32) - np.float32(0.5)
    return ops


class Layer:
    def __init__(self, items, item_size, extra_size):
        assert (items > 0)
        assert (item_size > 0)
        assert (extra_size >= 0)
        self.items = items
        self.item_size = item_size
        self.extra_size = extra_size

    def build(self):
        self._expand_op = np.zeros((self.items, self.items * self.item_size),
                                   dtype=np.float32)
        for i in range(self.items):
            self._expand_op[i, i * self.item_size:(i + 1) * self.item_size] = np.float32(1.0)

    def call(self, inputs, ops):
        op_mask_part = inputs[:self.items * self.item_size]
        if self.extra_size > 0:
            ext_part = inputs[self.items * self.item_size:]
        else:
            ext_part = None
        # if ops in [-0.5, 0.5] or [-0.5 .. 0.5]:
        ops1 = np.add(ops, float(0.5))  # optional
        extended_op = np.matmul(ops1, self._expand_op)
        if self.extra_size > 0:
            return np.concatenate((np.multiply(op_mask_part, extended_op), ext_part))
        else:
            return np.multiply(op_mask_part, extended_op)


class Network:
    def __init__(self, given_layers):
        self.layers = given_layers
        self.input_layer = given_layers[0]
        self.output_layer = given_layers[-1]

    def build(self):
        for item in self.layers:
            item.build()

    def call(self, input_data):
        ops = RandomOps(self.input_layer.items)
        layer_output = self.input_layer.call(input_data, ops)

        count = 0
        print(layer_output, " Layer id: ", count)

        for item in self.layers[1:]:
            ops = RandomOps(self.input_layer.items)
            layer_output = item.call(layer_output, ops)
            count += 1
            print(layer_output, " Layer id: ", count)

        return layer_output


def main():
    items = 5
    item_size = 1
    extra_size = 0

    layers = []

    for i in range(3):
        item = Layer(items=items, item_size=item_size, extra_size=extra_size)
        layers.append(item)

    test_network = Network(layers)
    test_network.build()

    user_input = list(map(float, input("Enter 5 numbers: ").split()))
    result = test_network.call(user_input)
    print(result)


if __name__ == '__main__':
    main()
