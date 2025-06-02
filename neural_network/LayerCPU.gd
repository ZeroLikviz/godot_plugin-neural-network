class_name LayerCPU

var neuron_activations: Array = []
var neuron_inputs: Array = []
var weights: Array = []
var biases: Array = []
var next_layer_neuron_count: int = 0
var activation_function: ExpressionFunction
var activation_derivative: ExpressionFunction
var neuron_data: Array = []
var weight_data: Array = []
var bias_data: Array = []
var neuron_data_indices: Dictionary = {}
var weight_data_indices: Dictionary = {}
var bias_data_indices: Dictionary = {}
var optimizer: Optimizer = null

func _generate_random_weights(count: int) -> Array:
	var random_weights := []
	random_weights.resize(count)
	for i in range(count):
		random_weights[i] = randf_range(-1.0, 1.0)
	return random_weights

func _init(current_neuron_count: int, next_neuron_count: int, use_bias: bool, activation: String, derivative: String = "auto") -> void:
	neuron_activations.resize(current_neuron_count)
	neuron_activations.fill(0.0)
	neuron_inputs = neuron_activations.duplicate()
	weights.resize(current_neuron_count)
	self.next_layer_neuron_count = next_neuron_count
	set_activation_function(activation, derivative)
	for i in range(weights.size()):
		weights[i] = _generate_random_weights(next_neuron_count)
	if use_bias:
		biases = _generate_random_weights(next_neuron_count)
	add_neuron_data("gradient")
	add_weight_data("gradient")
	add_bias_data("gradient")

func add_neuron_data(data_id: String) -> void:
	neuron_data_indices[data_id] = neuron_data.size()
	var new_data := []
	new_data.resize(neuron_activations.size())
	new_data.fill(0.0)
	neuron_data.append(new_data)

func add_weight_data(data_id: String) -> void:
	weight_data_indices[data_id] = weight_data.size()
	var new_weight_data := []
	new_weight_data.resize(neuron_activations.size())
	for i in range(neuron_activations.size()):
		new_weight_data[i] = []
		new_weight_data[i].resize(next_layer_neuron_count)
		new_weight_data[i].fill(0.0)
	weight_data.append(new_weight_data)

func add_bias_data(data_id: String) -> void:
	bias_data_indices[data_id] = bias_data.size()
	var new_bias_data := []
	new_bias_data.resize(biases.size())
	new_bias_data.fill(0.0)
	bias_data.append(new_bias_data)

func remove_bias_data(data_id: String) -> void:
	bias_data[bias_data_indices[data_id]].clear()
	bias_data_indices.erase(data_id)

func remove_weight_data(data_id: String) -> void:
	weight_data[weight_data_indices[data_id]].clear()
	weight_data_indices.erase(data_id)

func remove_neuron_data(data_id: String) -> void:
	neuron_data[neuron_data_indices[data_id]].clear()
	neuron_data_indices.erase(data_id)

func zero_gradients() -> void:
	neuron_data[neuron_data_indices["gradient"]].fill(0.0)
	for i in range(neuron_activations.size()):
		weight_data[weight_data_indices["gradient"]][i].fill(0.0)
	bias_data[bias_data_indices["gradient"]].fill(0.0)

func forward(next_layer: LayerCPU) -> void:
	if biases.size() > 0:
		for i in range(next_layer.neuron_inputs.size()):
			next_layer.neuron_inputs[i] = biases[i]
	else:
		next_layer.neuron_inputs.fill(0.0)
	for i in range(neuron_activations.size()):
		for j in range(next_layer.neuron_inputs.size()):
			next_layer.neuron_inputs[j] += neuron_activations[i] * weights[i][j]
	for i in range(next_layer.neuron_activations.size()):
		next_layer.neuron_activations[i] = next_layer.activation_function.callable.call(next_layer.neuron_inputs[i])

func backpropagate(next_layer: LayerCPU) -> void:
	var neuron_grad_idx: int = neuron_data_indices["gradient"]
	var weight_grad_idx: int = weight_data_indices["gradient"]
	var bias_grad_idx: int = bias_data_indices["gradient"]
	var next_neuron_grad_idx: int = next_layer.neuron_data_indices["gradient"]
	for i in range(neuron_activations.size()):
		for j in range(next_layer_neuron_count):
			var delta: float = next_layer.neuron_data[next_neuron_grad_idx][j]
			var derivative: float = next_layer.activation_derivative.callable.call(next_layer.neuron_inputs[j])
			weight_data[weight_grad_idx][i][j] = delta * derivative * neuron_activations[i]
	if biases.size() > 0:
		for j in range(next_layer_neuron_count):
			var delta: float = next_layer.neuron_data[next_neuron_grad_idx][j]
			var derivative: float = next_layer.activation_derivative.callable.call(next_layer.neuron_inputs[j])
			bias_data[bias_grad_idx][j] = delta * derivative
	for i in range(neuron_activations.size()):
		neuron_data[neuron_grad_idx][i] = 0.0
		for j in range(next_layer_neuron_count):
			var delta: float = next_layer.neuron_data[next_neuron_grad_idx][j]
			neuron_data[neuron_grad_idx][i] += delta * weights[i][j]
		neuron_data[neuron_grad_idx][i] *= activation_derivative.callable.call(neuron_inputs[i])

func set_activations(values: Array) -> void:
	for i in range(neuron_activations.size()):
		neuron_activations[i] = values[i]

func set_activation_function(function: String, derivative: String) -> void:
	activation_function = ExpressionFunction.new("func(x: float) -> float:\n", function)
	if NetworkConstants.ACTIVATION_TO_DERIVATIVE.has(function):
		activation_derivative = ExpressionFunction.new("func(x: float) -> float:\n", NetworkConstants.ACTIVATION_TO_DERIVATIVE[function].indent("\t"))
	elif derivative == "auto":
		derivative = "static var act: Callable = func(x: float) -> float:\n" + function + "\n" + \
		             "return (act.call(x + NetworkConstants.EPS) - act.call(x)) / NetworkConstants.EPS"
		activation_derivative = ExpressionFunction.new("func(x: float) -> float:\n", derivative)
	else:
		activation_derivative = ExpressionFunction.new("func(x: float) -> float:\n", derivative)

func set_optimizer(optimizer: Optimizer) -> void:
	self.optimizer = optimizer.duplicate()
	self.optimizer.initialize(self)

func apply_gradients() -> void:
	optimizer.apply_gradients(self)

func toggle_bias(enable: bool) -> void:
	if enable:
		biases = _generate_random_weights(next_layer_neuron_count)
	else:
		biases = []
	optimizer.initialize(self)

func size() -> int:
	return neuron_activations.size()
