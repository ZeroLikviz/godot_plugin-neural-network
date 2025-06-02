class_name NetworkCPU

var layers: Array[LayerCPU] = []
var loss_function: ExpressionFunction

func _init(layer_sizes: Array, use_bias: bool, activation_funcs: Variant = ActivationFunctions.Sigmoid, optimizers: Variant = AdamOptimizer.new()) -> void:
	if layer_sizes.size() < 2:
		push_error("Network requires at least 2 layers (input and output).")
		return
	for i in range(layer_sizes.size() - 1):
		layers.append(LayerCPU.new(layer_sizes[i], layer_sizes[i + 1], use_bias, ActivationFunctions.Sigmoid))
	layers.append(LayerCPU.new(layer_sizes[-1], 0, false, ActivationFunctions.Sigmoid))
	if activation_funcs is Array:
		if activation_funcs.size() != layers.size():
			push_error("Activation function array size must match layer count.")
			return
		for i in range(layers.size()):
			set_activation_function(i, activation_funcs[i])
	else:
		set_activation_functions(activation_funcs)
	if optimizers is Array:
		if optimizers.size() != layers.size():
			push_error("Optimizer array size must match layer count.")
			return
		for i in range(layers.size()):
			set_optimizer(i, optimizers[i])
	else:
		set_optimizers(optimizers)
	for layer in layers:
		layer.add_weight_data("batch_gradient")
		layer.add_bias_data("batch_gradient")
	set_loss_function(LossFunctions.MSE)

func insert_layer(index: int, neuron_count: int, use_bias: bool, activation: String = ActivationFunctions.Sigmoid, optimizer: Variant = null) -> void:
	if index < -1 or index >= layers.size():
		push_error("Invalid layer index: %d" % index)
		return
	if index == 0:
		layers.push_front(LayerCPU.new(neuron_count, layers[0].neuron_activations.size(), use_bias, activation))
		set_activation_function(0, activation)
		set_optimizer(0, optimizer)
	elif index == -1 or index == layers.size() - 1:
		var last_layer_size: int = layers[-1].neuron_activations.size()
		layers[-1] = LayerCPU.new(last_layer_size, neuron_count, use_bias, activation)
		layers.push_back(LayerCPU.new(neuron_count, 0, false,  ActivationFunctions.Sigmoid))
		set_activation_function(layers.size() - 2, activation)
		set_optimizer(layers.size() - 2, optimizer)
	else:
		layers.insert(index, LayerCPU.new(neuron_count, layers[index].neuron_activations.size(), use_bias, activation))
		if neuron_count != layers[index - 1].next_layer_neuron_count:
			change_layer(index - 1, layers[index - 1].neuron_activations.size(), neuron_count)
		set_activation_function(index, activation)
		set_optimizer(index, optimizer)

func change_layer(index: int, neuron_count: int, next_neuron_count: int) -> void:
	if index < 0 or index >= layers.size():
		push_error("Invalid layer index: %d" % index)
		return
	var optimizer: Optimizer = layers[index].optimizer
	layers[index] = LayerCPU.new(neuron_count, next_neuron_count, bool(layers[index].biases.size()), layers[index].activation_function.expression)
	set_optimizer(index, optimizer)

func remove_layer(index: int) -> void:
	if layers.size() <= 2:
		push_error("Cannot remove layer: network must have at least 2 layers.")
		return
	if index < -1 or index >= layers.size():
		push_error("Invalid layer index: %d" % index)
		return
	if index == 0:
		layers.remove_at(0)
	elif index == -1 or index == layers.size() - 1:
		layers.remove_at(layers.size() - 1)
		change_layer(layers.size() - 1, layers[-1].neuron_activations.size(), 0)
	else:
		layers.remove_at(index)
		change_layer(index - 1, layers[index - 1].neuron_activations.size(), layers[index].neuron_activations.size())

func set_loss_function(loss_func: String) -> void:
	loss_function = ExpressionFunction.new("func(predicted: Array, target: Array) -> float:\n", loss_func)

func set_activation_function(layer_idx: int, activation: String, derivative: String = "auto") -> void:
	if layer_idx < 0 or layer_idx >= layers.size():
		push_error("Invalid layer index: %d" % layer_idx)
		return
	else:
		layers[layer_idx].set_activation_function(activation, derivative)

func set_activation_functions(activation: String, derivative: String = "auto") -> void:
	for i in range(layers.size()):
		set_activation_function(i, activation, derivative)

func set_optimizer(layer_idx: int, optimizer: Optimizer) -> void:
	if layer_idx < 0 or layer_idx >= layers.size():
		push_error("Invalid layer index: %d" % layer_idx)
		return
	else:
		layers[layer_idx].set_optimizer(optimizer)

func set_optimizers(optimizer: Optimizer) -> void:
	for i in range(layers.size()):
		set_optimizer(i, optimizer)

func toggle_bias(layer_idx: int, enable: bool) -> void:
	if layer_idx < 0 or layer_idx >= layers.size():
		push_error("Invalid layer index: %d" % layer_idx)
		return
	layers[layer_idx].toggle_bias(enable)

func toggle_biases(enable: bool) -> void:
	for layer in layers:
		layer.toggle_bias(enable)

func forward(input: Array) -> Array:
	if input.size() != layers[0].neuron_activations.size():
		push_error("Input size (%d) does not match input layer size (%d)." % [input.size(), layers[0].neuron_activations.size()])
		return []
	layers[0].set_activations(input)
	for i in range(layers.size() - 1):
		layers[i].forward(layers[i + 1])
	return layers[-1].neuron_activations.duplicate()

func predict(input: Array) -> Array:
	if input.size() != layers[0].neuron_activations.size():
		push_error("Input size (%d) does not match input layer size (%d)." % [input.size(), layers[0].neuron_activations.size()])
		return []
	layers[0].set_activations(input)
	for i in range(layers.size() - 1):
		layers[i].forward(layers[i + 1])
	return layers[-1].neuron_activations.duplicate()

func backpropagate(target: Array) -> void:
	if target.size() != layers[-1].neuron_activations.size():
		push_error("Target size (%d) does not match output layer size (%d)." % [target.size(), layers[-1].neuron_activations.size()])
		return
	var derivative_idx: int = layers[-1].neuron_data_indices["gradient"]
	for i in range(layers[-1].neuron_activations.size()):
		var original_value: float = layers[-1].neuron_activations[i]
		var standard_loss: float = loss_function.callable.call(layers[-1].neuron_activations, target)
		layers[-1].neuron_activations[i] += NetworkConstants.EPS
		var changed_loss: float = loss_function.callable.call(layers[-1].neuron_activations, target)
		layers[-1].neuron_data[derivative_idx][i] = (changed_loss - standard_loss) / NetworkConstants.EPS
		layers[-1].neuron_activations[i] = original_value
	for i in range(layers.size() - 2, -1, -1):
		layers[i].backpropagate(layers[i + 1])

func _apply_gradients() -> void:
	for layer in layers:
		layer.apply_gradients()

func _clear_batch_gradients() -> void:
	for i in range(layers.size() - 1):
		var batch_weight_grad_idx: int = layers[i].weight_data_indices["batch_gradient"]
		var batch_bias_grad_idx: int = layers[i].bias_data_indices["batch_gradient"]
		for j in range(layers[i].size()):
			for k in range(layers[i + 1].size()):
				layers[i].weight_data[batch_weight_grad_idx][j][k] = 0.0
		if layers[i].biases.size() > 0:
			for j in range(layers[i + 1].size()):
				layers[i].bias_data[batch_bias_grad_idx][j] = 0.0

func _accumulate_batch_gradients() -> void:
	for i in range(layers.size() - 1):
		var batch_weight_grad_idx: int = layers[i].weight_data_indices["batch_gradient"]
		var batch_bias_grad_idx: int = layers[i].bias_data_indices["batch_gradient"]
		var weight_grad_idx: int = layers[i].weight_data_indices["gradient"]
		var bias_grad_idx: int = layers[i].bias_data_indices["gradient"]
		for j in range(layers[i].size()):
			for k in range(layers[i + 1].size()):
				layers[i].weight_data[batch_weight_grad_idx][j][k] += layers[i].weight_data[weight_grad_idx][j][k]
		if layers[i].biases.size() > 0:
			for j in range(layers[i + 1].size()):
				layers[i].bias_data[batch_bias_grad_idx][j] += layers[i].bias_data[bias_grad_idx][j]

func _average_batch_gradients(divisor: float) -> void:
	for i in range(layers.size() - 1):
		var batch_weight_grad_idx: int = layers[i].weight_data_indices["batch_gradient"]
		var batch_bias_grad_idx: int = layers[i].bias_data_indices["batch_gradient"]
		for j in range(layers[i].size()):
			for k in range(layers[i + 1].size()):
				layers[i].weight_data[batch_weight_grad_idx][j][k] /= divisor
		if layers[i].biases.size() > 0:
			for j in range(layers[i + 1].size()):
				layers[i].bias_data[batch_bias_grad_idx][j] /= divisor

func _transfer_batch_gradients() -> void:
	for i in range(layers.size() - 1):
		var batch_weight_grad_idx: int = layers[i].weight_data_indices["batch_gradient"]
		var batch_bias_grad_idx: int = layers[i].bias_data_indices["batch_gradient"]
		var weight_grad_idx: int = layers[i].weight_data_indices["gradient"]
		var bias_grad_idx: int = layers[i].bias_data_indices["gradient"]
		for j in range(layers[i].size()):
			for k in range(layers[i + 1].size()):
				layers[i].weight_data[weight_grad_idx][j][k] = layers[i].weight_data[batch_weight_grad_idx][j][k]
		if layers[i].biases.size() > 0:
			for j in range(layers[i + 1].size()):
				layers[i].bias_data[bias_grad_idx][j] = layers[i].bias_data[batch_bias_grad_idx][j]

func train(inputs: Array, targets: Array, batch_size: int = -1) -> void:
	if inputs.size() != targets.size():
		push_error("Inputs and targets arrays must have the same size.")
		return
	var indices: Array = range(inputs.size())
	if batch_size > 0 and batch_size < inputs.size():
		indices.shuffle()
		indices.resize(batch_size)
	_clear_batch_gradients()
	for i in indices:
		forward(inputs[i])
		backpropagate(targets[i])
		_accumulate_batch_gradients()
	_average_batch_gradients(indices.size())
	_transfer_batch_gradients()
	_apply_gradients()

func compute_loss(inputs: Array, targets: Array) -> float:
	if inputs.size() != targets.size():
		push_error("Inputs and targets arrays must have the same size.")
		return 0.0
	var total_loss: float = 0.0
	for i in range(inputs.size()):
		total_loss += loss_function.callable.call(forward(inputs[i]), targets[i])
	return total_loss / inputs.size()

static func placeholder() -> NetworkCPU:
	var network : NetworkCPU = NetworkCPU.new([1,1], false)
	return network
