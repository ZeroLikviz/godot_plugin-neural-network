## Resilient Propagation (RProp) optimizer.
extends Optimizer
class_name RpropOptimizer

## Increase factor for step size.
var eta_plus: float
## Decrease factor for step size.
var eta_minus: float
## Minimum step size.
var delta_min: float
## Maximum step size.
var delta_max: float
## Current step sizes for weights.
var delta: Array[PackedFloat32Array] = []
## Previous gradients for weights.
var prev_grad: Array[PackedFloat32Array] = []
## Current step sizes for biases.
var delta_bias: PackedFloat32Array = []
## Previous gradients for biases.
var prev_grad_bias: PackedFloat32Array = []

## [param ep] Increase factor for step size.
## [param em] Decrease factor for step size.
## [param dmin] Minimum step size.
## [param dmax] Maximum step size.
func _init(ep: float = 1.2, em: float = 0.5, dmin: float = 1e-6, dmax: float = 50.0) -> void:
	eta_plus = ep
	eta_minus = em
	delta_min = dmin
	delta_max = dmax

func initialize(layer: LayerCPU) -> void:
	delta.resize(layer.neuron_activations.size())
	prev_grad.resize(layer.neuron_activations.size())
	for i in range(layer.neuron_activations.size()):
		delta[i] = PackedFloat32Array()
		delta[i].resize(layer.next_layer_neuron_count)
		delta[i].fill(0.01)
		prev_grad[i] = PackedFloat32Array()
		prev_grad[i].resize(layer.next_layer_neuron_count)
		prev_grad[i].fill(0.0)
	delta_bias.resize(layer.biases.size())
	delta_bias.fill(0.01)
	prev_grad_bias.resize(layer.biases.size())
	prev_grad_bias.fill(0.0)

func apply_gradients(layer: LayerCPU) -> void:
	var grad_index: int = layer.weight_data_indices["gradient"]
	var bias_grad_index: int = layer.bias_data_indices["gradient"]
	for i in range(layer.neuron_activations.size()):
		for j in range(layer.next_layer_neuron_count):
			var grad: float = layer.weight_data[grad_index][i][j]
			var sign: float = prev_grad[i][j] * grad
			if sign > 0:
				delta[i][j] = min(delta[i][j] * eta_plus, delta_max)
			elif sign < 0:
				delta[i][j] = max(delta[i][j] * eta_minus, delta_min)
				grad = 0.0
			var grad_sign: float = int(grad > 0.0) - int(grad < 0.0)
			layer.weights[i][j] -= grad_sign * delta[i][j]
			prev_grad[i][j] = grad
	if layer.biases.size() == 0:
		return
	for j in range(layer.next_layer_neuron_count):
		var grad: float = layer.bias_data[bias_grad_index][j]
		var sign: float = prev_grad_bias[j] * grad
		if sign > 0:
			delta_bias[j] = min(delta_bias[j] * eta_plus, delta_max)
		elif sign < 0:
			delta_bias[j] = max(delta_bias[j] * eta_minus, delta_min)
			grad = 0.0
		layer.biases[j] -= sign(grad) * delta_bias[j]
		prev_grad_bias[j] = grad

func duplicate() -> Optimizer:
	return RpropOptimizer.new(eta_plus, eta_minus, delta_min, delta_max)

func get_class_name() -> String:
	return "RpropOptimizer"

func get_hyperparameters() -> Array:
	return [eta_plus, eta_minus, delta_min, delta_max]

func load_hyperparameters(hyperparameters: Array) -> void:
	eta_plus = hyperparameters[0]
	eta_minus = hyperparameters[1]
	delta_min = hyperparameters[2]
	delta_max = hyperparameters[3]
