## Adadelta optimizer.
extends Optimizer
class_name AdadeltaOptimizer

## Decay rate for running averages.
var rho: float
## Small constant for numerical stability.
var epsilon: float
## Running average of squared gradients for weights.
var g2: Array[PackedFloat32Array] = []
## Running average of squared updates for weights.
var d2: Array[PackedFloat32Array] = []
## Running average of squared gradients for biases.
var g2_bias: PackedFloat32Array = []
## Running average of squared updates for biases.
var d2_bias: PackedFloat32Array = []

## [param r] Decay rate for running averages.
## [param eps] Epsilon for numerical stability.
func _init(r: float = 0.95, eps: float = 1e-6) -> void:
	rho = r
	epsilon = eps

func initialize(layer: LayerCPU) -> void:
	g2.resize(layer.neuron_activations.size())
	d2.resize(layer.neuron_activations.size())
	for i in range(layer.neuron_activations.size()):
		g2[i] = PackedFloat32Array()
		g2[i].resize(layer.next_layer_neuron_count)
		g2[i].fill(0.0)
		d2[i] = PackedFloat32Array()
		d2[i].resize(layer.next_layer_neuron_count)
		d2[i].fill(0.0)
	g2_bias.resize(layer.biases.size())
	g2_bias.fill(0.0)
	d2_bias.resize(layer.biases.size())
	d2_bias.fill(0.0)

func apply_gradients(layer: LayerCPU) -> void:
	var grad_index: int = layer.weight_data_indices["gradient"]
	var bias_grad_index: int = layer.bias_data_indices["gradient"]
	for i in range(layer.neuron_activations.size()):
		for j in range(layer.next_layer_neuron_count):
			var grad: float = layer.weight_data[grad_index][i][j]
			g2[i][j] = rho * g2[i][j] + (1.0 - rho) * grad * grad
			var delta: float = sqrt(d2[i][j] + epsilon) / sqrt(g2[i][j] + epsilon) * grad
			d2[i][j] = rho * d2[i][j] + (1.0 - rho) * delta * delta
			layer.weights[i][j] -= delta
	if layer.biases.size() == 0:
		return
	for j in range(layer.next_layer_neuron_count):
		var grad: float = layer.bias_data[bias_grad_index][j]
		g2_bias[j] = rho * g2_bias[j] + (1.0 - rho) * grad * grad
		var delta: float = sqrt(d2_bias[j] + epsilon) / sqrt(g2_bias[j] + epsilon) * grad
		d2_bias[j] = rho * d2_bias[j] + (1.0 - rho) * delta * delta
		layer.biases[j] -= delta

func duplicate() -> Optimizer:
	return AdadeltaOptimizer.new(rho, epsilon)

func get_class_name() -> String:
	return "AdadeltaOptimizer"

func get_hyperparameters() -> Array:
	return [rho, epsilon]

func load_hyperparameters(hyperparameters: Array) -> void:
	rho = hyperparameters[0]
	epsilon = hyperparameters[1]
