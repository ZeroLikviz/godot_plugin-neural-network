# Represents a single layer in a neural network, handling neuron data, weights, biases, and operations.
class_name LayerCPU

# --- Layer Properties ---

# Stores neuron activation values after applying the activation function.
var neuron_activations: Array = []

# Stores pre-activation input values for neurons.
var neuron_inputs: Array = []

# Stores weights connecting this layer to the next. weights[i][j] represents the weight from neuron i in this layer to neuron j in the next.
var weights: Array = []

# Stores biases for neurons in the next layer.
var biases: Array = []

# Number of neurons in the next layer.
var next_layer_neuron_count: int = 0

# Callable function applied to the next layer's neuron inputs to compute activations.
var activation_function: ExpressionFunction

# Derivative of the activation function, used for backpropagation.
var activation_derivative: ExpressionFunction

# Stores additional neuron data (e.g., gradients). Each element is an Array for a specific data type.
var neuron_data: Array = []

# Stores additional weight data (e.g., gradients). weight_data[k][i][j] holds data for the weight from neuron i to neuron j in the next layer.
var weight_data: Array = []

# Stores additional bias data (e.g., gradients). bias_data[k][j] holds data for the bias of neuron j in the next layer.
var bias_data: Array = []

# Maps data type names to indices in neuron_data for quick access.
var neuron_data_indices: Dictionary = {}

# Maps data type names to indices in weight_data for quick access.
var weight_data_indices: Dictionary = {}

# Maps data type names to indices in bias_data for quick access.
var bias_data_indices: Dictionary = {}

# Optimizer instance for updating weights and biases during training.
var optimizer: Optimizer = null

# --- Utility Functions ---

# Generates an array of random weights in the range [-1.0, 1.0].
# @param count Number of weights to generate.
# @return An Array containing the generated random weights.
func _generate_random_weights(count: int) -> Array:
	var random_weights := []
	random_weights.resize(count)
	for i in range(count):
		random_weights[i] = randf_range(-1.0, 1.0)
	return random_weights

# --- Initialization ---

# Initializes the layer with specified neuron counts, biases, and activation function.
# @param current_neuron_count Number of neurons in this layer.
# @param next_neuron_count Number of neurons in the next layer.
# @param use_bias Whether to include biases for the next layer.
# @param activation Activation function for the next layer (Callable, String, or default identity function).
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

# --- Data Management ---

# Adds storage for neuron-specific data, such as gradients, with a unique identifier.
# @param data_id Unique identifier for the data type, used as a key in neuron_data_indices.
func add_neuron_data(data_id: String) -> void:
	neuron_data_indices[data_id] = neuron_data.size()
	var new_data := []
	new_data.resize(neuron_activations.size())
	new_data.fill(0.0)
	neuron_data.append(new_data)

# Adds storage for weight-specific data, such as gradients, with a unique identifier.
# @param data_id Unique identifier for the data type, used as a key in weight_data_indices.
func add_weight_data(data_id: String) -> void:
	weight_data_indices[data_id] = weight_data.size()
	var new_weight_data := []
	new_weight_data.resize(neuron_activations.size())
	for i in range(neuron_activations.size()):
		new_weight_data[i] = []
		new_weight_data[i].resize(next_layer_neuron_count)
		new_weight_data[i].fill(0.0)
	weight_data.append(new_weight_data)

# Adds storage for bias-specific data, such as gradients, with a unique identifier.
# @param data_id Unique identifier for the data type, used as a key in bias_data_indices.
func add_bias_data(data_id: String) -> void:
	bias_data_indices[data_id] = bias_data.size()
	var new_bias_data := []
	new_bias_data.resize(biases.size())
	new_bias_data.fill(0.0)
	bias_data.append(new_bias_data)

# Removes bias-specific data storage for the specified identifier.
# @param data_id Identifier of the bias data to remove.
func remove_bias_data(data_id: String) -> void:
	bias_data[bias_data_indices[data_id]].clear()
	bias_data_indices.erase(data_id)

# Removes weight-specific data storage for the specified identifier.
# @param data_id Identifier of the weight data to remove.
func remove_weight_data(data_id: String) -> void:
	weight_data[weight_data_indices[data_id]].clear()
	weight_data_indices.erase(data_id)

# Removes neuron-specific data storage for the specified identifier.
# @param data_id Identifier of the neuron data to remove.
func remove_neuron_data(data_id: String) -> void:
	neuron_data[neuron_data_indices[data_id]].clear()
	neuron_data_indices.erase(data_id)

# Resets all gradient data (neuron, weight, and bias) to zero.
func zero_gradients() -> void:
	neuron_data[neuron_data_indices["gradient"]].fill(0.0)
	for i in range(neuron_activations.size()):
		weight_data[weight_data_indices["gradient"]][i].fill(0.0)
	bias_data[bias_data_indices["gradient"]].fill(0.0)

# --- Neural Network Operations ---

# Performs forward propagation to compute the next layer's neuron activations.
# @param next_layer The next LayerCPU instance in the network.
func forward(next_layer: LayerCPU) -> void:
	# Initialize next layer's neuron inputs with biases (if any)
	if biases.size() > 0:
		for i in range(next_layer.neuron_inputs.size()):
			next_layer.neuron_inputs[i] = biases[i]
	else:
		next_layer.neuron_inputs.fill(0.0)
	
	# Compute weighted sum for next layer's inputs
	for i in range(neuron_activations.size()):
		for j in range(next_layer.neuron_inputs.size()):
			next_layer.neuron_inputs[j] += neuron_activations[i] * weights[i][j]
	
	# Apply activation function to compute next layer's activations
	for i in range(next_layer.neuron_activations.size()):
		next_layer.neuron_activations[i] = next_layer.activation_function.callable.call(next_layer.neuron_inputs[i])

# Performs backpropagation to compute gradients for weights, biases, and neurons, assuming next layer's gradients are computed.
# @param next_layer The next LayerCPU instance in the network.
func backpropagate(next_layer: LayerCPU) -> void:
	var neuron_grad_idx: int = neuron_data_indices["gradient"]
	var weight_grad_idx: int = weight_data_indices["gradient"]
	var bias_grad_idx: int = bias_data_indices["gradient"]
	var next_neuron_grad_idx: int = next_layer.neuron_data_indices["gradient"]
	
	# Compute gradients for weights
	for i in range(neuron_activations.size()):
		for j in range(next_layer_neuron_count):
			var delta: float = next_layer.neuron_data[next_neuron_grad_idx][j]
			var derivative: float = next_layer.activation_derivative.callable.call(next_layer.neuron_inputs[j])
			weight_data[weight_grad_idx][i][j] = delta * derivative * neuron_activations[i]
	
	# Compute gradients for biases (if used)
	if biases.size() > 0:
		for j in range(next_layer_neuron_count):
			var delta: float = next_layer.neuron_data[next_neuron_grad_idx][j]
			var derivative: float = next_layer.activation_derivative.callable.call(next_layer.neuron_inputs[j])
			bias_data[bias_grad_idx][j] = delta * derivative
	
	# Compute gradients for neurons
	for i in range(neuron_activations.size()):
		neuron_data[neuron_grad_idx][i] = 0.0
		for j in range(next_layer_neuron_count):
			var delta: float = next_layer.neuron_data[next_neuron_grad_idx][j]
			neuron_data[neuron_grad_idx][i] += delta * weights[i][j]
		neuron_data[neuron_grad_idx][i] *= activation_derivative.callable.call(neuron_inputs[i])

# Sets the activation values for this layer's neurons.
# @param values Array of activation values, must match the layer's neuron count.
func set_activations(values: Array) -> void:
	for i in range(neuron_activations.size()):
		neuron_activations[i] = values[i]

# Sets the activation function and its derivative for the layer.
# @param function Activation function as a Callable, GDScript string, or ActivationFunctions instance.
# @param derivative Optional derivative as a Callable or GDScript string; if null, approximated numerically.
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

# Sets the optimizer for updating weights and biases during training.
# @param optimizer_class GDScript class defining the optimizer.
func set_optimizer(optimizer: Optimizer) -> void:
	self.optimizer = optimizer.duplicate()
	self.optimizer.initialize(self)

# Applies gradients using the current optimizer to update weights and biases.
func apply_gradients() -> void:
	optimizer.apply_gradients(self)

# Enables or disables biases for the layer and reinitializes the optimizer.
# @param enable True to enable biases, False to disable and clear biases.
func toggle_bias(enable: bool) -> void:
	if enable:
		biases = _generate_random_weights(next_layer_neuron_count)
	else:
		biases = []
	optimizer.initialize(self)

# Returns number of neurons.
func size() -> int:
	return neuron_activations.size()
