## Yogi optimizer.
extends Optimizer
class_name YogiOptimizer

## Learning rate for updates.
var learning_rate: float
## Exponential decay rate for first moment.
var beta1: float
## Exponential decay rate for second moment.
var beta2: float
## Small constant for numerical stability.
var epsilon: float
## First moment (mean) for weights.
var m: Array[PackedFloat32Array] = []
## Second moment (variance) for weights.
var v: Array[PackedFloat32Array] = []
## First moment for biases.
var m_bias: PackedFloat32Array = []
## Second moment for biases.
var v_bias: PackedFloat32Array = []
## Timestep counter.
var t: int = 0

## [param lr] Learning rate.
## [param b1] Beta1 (first moment decay).
## [param b2] Beta2 (second moment decay).
## [param eps] Epsilon for numerical stability.
func _init(lr: float = 0.01, b1: float = 0.9, b2: float = 0.999, eps: float = NetworkConstants.EPS) -> void:
	learning_rate = lr
	beta1 = b1
	beta2 = b2
	epsilon = eps

func initialize(layer: LayerCPU) -> void:
	m.resize(layer.neuron_activations.size())
	v.resize(layer.neuron_activations.size())
	for i in range(layer.neuron_activations.size()):
		m[i] = PackedFloat32Array()
		m[i].resize(layer.next_layer_neuron_count)
		m[i].fill(0.0)
		v[i] = PackedFloat32Array()
		v[i].resize(layer.next_layer_neuron_count)
		v[i].fill(0.0)
	m_bias.resize(layer.biases.size())
	m_bias.fill(0.0)
	v_bias.resize(layer.biases.size())
	v_bias.fill(0.0)

func apply_gradients(layer: LayerCPU) -> void:
	var grad_index: int = layer.weight_data_indices["gradient"]
	var bias_grad_index: int = layer.bias_data_indices["gradient"]
	t += 1
	for i in range(layer.neuron_activations.size()):
		for j in range(layer.next_layer_neuron_count):
			var grad: float = layer.weight_data[grad_index][i][j]
			var g2 := grad * grad
			m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * grad
			var sign_term := sign(v[i][j] - g2)
			v[i][j] -= (1.0 - beta2) * sign_term * g2
			var m_hat: float = m[i][j] / (1.0 - pow(beta1, t))
			var v_hat: float = v[i][j] / (1.0 - pow(beta2, t))
			layer.weights[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
	if layer.biases.size():
		return
	for j in range(layer.next_layer_neuron_count):
		var grad: float = layer.bias_data[bias_grad_index][j]
		var g2 := grad * grad
		m_bias[j] = beta1 * m_bias[j] + (1.0 - beta1) * grad
		var sign_term := sign(v_bias[j] - g2)
		v_bias[j] -= (1.0 - beta2) * sign_term * g2
		var m_hat: float = m_bias[j] / (1.0 - pow(beta1, t))
		var v_hat: float = v_bias[j] / (1.0 - pow(beta2, t))
		layer.biases[j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)

func duplicate() -> Optimizer:
	return YogiOptimizer.new(learning_rate, beta1, beta2, epsilon)

func get_class_name() -> String:
	return "YogiOptimizer"

func get_hyperparameters() -> Array:
	return [learning_rate, beta1, beta2, epsilon]

func load_hyperparameters(hyperparameters: Array) -> void:
	learning_rate = hyperparameters[0]
	beta1 = hyperparameters[1]
	beta2 = hyperparameters[2]
	epsilon = hyperparameters[3]
