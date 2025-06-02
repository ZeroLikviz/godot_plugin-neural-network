## Gradient Descent optimizer.
extends Optimizer
class_name GradientDescentOptimizer

## Learning rate for weight and bias updates.
var learning_rate: float

## [param lr] Learning rate.
func _init(lr: float = 0.01) -> void:
	learning_rate = lr

func apply_gradients(layer: LayerCPU) -> void:
	var grad_index: int = layer.weight_data_indices["gradient"]
	var bias_grad_index: int = layer.bias_data_indices["gradient"]
	for i in range(layer.neuron_activations.size()):
		for j in range(layer.next_layer_neuron_count):
			layer.weights[i][j] -= learning_rate * layer.weight_data[grad_index][i][j]
	if layer.biases.size() == 0:
		return
	for j in range(layer.next_layer_neuron_count):
		layer.biases[j] -= learning_rate * layer.bias_data[bias_grad_index][j]

func duplicate() -> Optimizer:
	return GradientDescentOptimizer.new(learning_rate)

func get_class_name() -> String:
	return "GradientDescentOptimizer"

func get_hyperparameters() -> Array:
	return [learning_rate]

func load_hyperparameters(hyperparameters: Array) -> void:
	learning_rate = hyperparameters[0]
