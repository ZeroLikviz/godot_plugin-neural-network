## Nesterov Accelerated Gradient (NAG) optimizer.
extends Optimizer
class_name NAGOptimizer

## Learning rate for updates.
var learning_rate: float
## Momentum factor.
var gamma: float
## Velocity for weights.
var v: Array[PackedFloat32Array] = []
## Velocity for biases.
var v_bias: PackedFloat32Array = []

## [param lr] Learning rate.
## [param g] Momentum factor.
func _init(lr: float = 0.01, g: float = 0.9) -> void:
	learning_rate = lr
	gamma = g

func initialize(layer: LayerCPU) -> void:
	v.resize(layer.neuron_activations.size())
	for i in range(layer.neuron_activations.size()):
		v[i] = PackedFloat32Array()
		v[i].resize(layer.next_layer_neuron_count)
		v[i].fill(0.0)
	v_bias.resize(layer.biases.size())
	v_bias.fill(0.0)

func apply_gradients(layer: LayerCPU) -> void:
	var grad_index: int = layer.weight_data_indices["gradient"]
	var bias_grad_index: int = layer.bias_data_indices["gradient"]
	for i in range(layer.neuron_activations.size()):
		for j in range(layer.next_layer_neuron_count):
			var grad: float = layer.weight_data[grad_index][i][j]
			var v_prev = v[i][j]
			v[i][j] = gamma * v[i][j] + learning_rate * grad
			layer.weights[i][j] -= gamma * v[i][j] + (1 - gamma) * v_prev
	if layer.biases.size() == 0:
		return
	for j in range(layer.next_layer_neuron_count):
		var grad: float = layer.bias_data[bias_grad_index][j]
		v_bias[j] = gamma * v_bias[j] + learning_rate * grad
		layer.biases[j] -= gamma * v_bias[j] + learning_rate * grad

func duplicate() -> Optimizer:
	return NAGOptimizer.new(learning_rate, gamma)

func get_class_name() -> String:
	return "NAGOptimizer"

func get_hyperparameters() -> Array:
	return [learning_rate, gamma]

func load_hyperparameters(hyperparameters: Array) -> void:
	learning_rate = hyperparameters[0]
	gamma = hyperparameters[1]
