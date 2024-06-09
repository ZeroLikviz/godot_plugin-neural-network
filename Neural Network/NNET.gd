@icon("res://addons/Neural Network/base NNET class.gd")
extends BaseNNET
class_name NNET

var structure : Array = []
var weights : Array = []
var neurons_in : Array = []
var neurons_out : Array = []
var biases : Array = []
var deltas : Array = []
var target : Array = [] # desired output
var signs : Array = []
var moments : Array = []
var smoments : Array = []
var bias_moments : Array = [] # moments for biases
var bias_smoments : Array = [] # smoments for biases
var bias_signs : Array = []
var active_neurons : Array = []
var current_function : BaseNNET.ActivationFunctions
var user_functions : Array[Callable] = []
var f : Array[Callable] = []
var df : Array[Callable] = []
var bias : bool # use bias
var lr : float # learning rate
var p : float # probability for dropout
var layer_der : Array = [] # layer derevatives
var get_bias : Callable
var setup_resilient_propagation : Callable
var allocate_target : Callable
var max_step : float
var min_step : float
var update_value : float
var multiplication_factor : float
var reduction_factor : float
var algorithm : BaseNNET.Algorithm
var loss : Callable
var layer_ratio : Callable # for dropout algorithm
var dropout_neurons : Callable
var bias_consts : Array = [] # adding softmax avoiding performance issues required serious changes, and this variable is one of the consequences of adding softmax. :P
var beta1 : float # for Adam
var beta2 : float # for Adam
var weight_decay : float # for Adam


func _init(new_structure : Array, learning_rate : float, use_bias : bool) -> void:
	if not is_structure_valid(new_structure):
		print("Structure size must be 2 or above. All elements of the structure must be integers. Example: [1,3,8,2]")
	bias = use_bias
	structure = new_structure.duplicate()
	init_callables()
	use_backpropogation(learning_rate)
	allocate_deltas()
	allocate_neurons()
	allocate_weights()
	allocate_biases()
	init_active_neurons()
	allocate_bias_consts()
	make_all_neurons_active()
	init_functions(BaseNNET.ActivationFunctions.sigmoid)
	set_loss_function(BaseNNET.LossFunctions.Cosine_similarity_loss)

func init_callables() -> void:
	if bias:
		get_bias = func(layer, neuron) -> float:
			return biases[layer - 1][neuron]
	else:
		get_bias = func(_layer, _neuron) -> float:
			return 0.0
	init_setup_resilient_propagation()
	allocate_target = func() -> void:
		target = []; target.resize(structure[structure.size() - 1])
		target.fill(0.0)
		allocate_target = func() -> void: pass
	f.resize(structure.size())
	df.resize(structure.size())
	user_functions.resize(structure.size())
	dropout_neurons = func() -> void: pass
	layer_ratio = func(_layer : int) -> float : return 1.0

func init_setup_resilient_propagation() -> void:
	setup_resilient_propagation = func() -> void:
		run()
		find_deltas()
		fill_signs()
		setup_resilient_propagation = func() -> void: pass

func allocate_neurons() -> void:
	allocate_neurons_like_structure(neurons_out)
	allocate_neurons_like_structure(neurons_in)
	neurons_in[0] = [1.0]

func allocate_neurons_like_structure(array : Array) -> void:
	allocate_target.call()
	array.resize(structure.size())
	var i : int = 0
	while i < structure.size():
		array[i] = []
		array[i].resize(structure[i])
		array[i].fill(0.0)
		i += 1

func allocate_biases() -> void:
	if bias: allocate_biases_like_structure(biases)

func allocate_biases_like_structure(array : Array) -> void:
	array.resize(structure.size() - 1)
	var i : int = 0
	while i < structure.size() - 1:
		array[i] = []
		array[i].resize(structure[i + 1])
		var j : int = 0
		while j < array[i].size():
			array[i][j] = randf_range(-3.0, 3.0)
			j += 1
		i += 1

func allocate_deltas() -> void:
	deltas.resize(structure.size() - 1)
	var i : int = 0
	while i < deltas.size():
		deltas[i] = []
		deltas[i].resize(structure[i + 1])
		deltas[i].fill(0.0)
		i += 1

func allocate_weights() -> void:
	allocate_weights_like_structure(weights)

func allocate_weights_like_structure(array : Array) -> void:
	array.resize(structure.size() - 1)
	var i : int = 0
	var j : int = 0
	var k : int = 0
	while i < structure.size() - 1:
		array[i] = []
		array[i].resize(structure[i])
		j = 0
		while j < structure[i]:
			array[i][j] = []
			array[i][j].resize(structure[i + 1])
			k = 0
			while k < array[i][j].size():
				array[i][j][k] = randf_range(-3.0, 3.0)
				k += 1
			j += 1
		i += 1

func allocate_bias_consts() -> void:
	bias_consts.resize(structure.size())
	bias_consts.fill(1.0)

func set_function(function : Variant, layer : int) -> void:
	set_specific_function(f, df, layer, function)

func set_specific_function(callable : Array, dcallable : Array, pos : int, function : Variant) -> void:
	if function is Callable:
		user_functions[pos] = function
		f[pos] = func(layer : int) -> void:
			var i : int = 0; while i < structure[layer]:
				neurons_out[layer][i] = user_functions[pos].call(neurons_in[layer][i])
				i += 1
		df[pos] = func(layer : int) -> void:
			layer_der.resize(structure[layer])
			var i : int = 0; while i < structure[layer]:
				layer_der[i] = (neurons_out[layer][i] \
				- user_functions[pos].call(neurons_in[layer][i] + BaseNNET.apzero)) / BaseNNET.apzero
				i += 1
	elif function is BaseNNET.ActivationFunctions:
		current_function = function
		match function:
			BaseNNET.ActivationFunctions.identity:
				callable[pos] = func(layer : int) -> void:
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = neurons_in[layer][i]
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var i : int = 0; while i < structure[layer]:
						layer_der[i] = 1.0
						i += 1
				bias_consts[pos] = 1.0
			BaseNNET.ActivationFunctions.binary_step:
				callable[pos] = func(layer : int) -> void:
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = float(neurons_in[layer][i] >= 0)
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var i : int = 0; while i < structure[layer]:
						layer_der[i] = 0.0
						i += 1
				bias_consts[pos] = 0.0
			BaseNNET.ActivationFunctions.logistic:
				callable[pos] = func(layer : int) -> void:
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = 1.0 / ( 1.0 + pow(2.7182, -neurons_in[layer][i]) )
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var i : int = 0; while i < structure[layer]:
						layer_der[i] = neurons_out[layer][i] * (1.0 - neurons_out[layer][i])
						i += 1
				bias_consts[pos] = -4.67041
			BaseNNET.ActivationFunctions.tanh:
				callable[pos] = func(layer : int) -> void:
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = 1.0 / ( 1.0 + pow(2.7182, -neurons_in[layer][i]) ) * 2 - 1.0
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var i : int = 0; while i < structure[layer]:
						layer_der[i] = 1.0 - neurons_out[layer][i] * neurons_out[layer][i]
						i += 1
				bias_consts[pos] = 0.462104
			BaseNNET.ActivationFunctions.ReLU:
				callable[pos] = func(layer : int) -> void:
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = neurons_in[layer][i] * int(neurons_in[layer][i] > 0)
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var i : int = 0; while i < structure[layer]:
						layer_der[i] = float(neurons_in[layer][i] > 0)
						i += 1
				bias_consts[pos] = 1.0
			BaseNNET.ActivationFunctions.mish:
				callable[pos] = func(layer : int) -> void:
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = (1.0 / ( 1.0 + pow(2.7182, -log( 1.0 + pow(2.7182, neurons_in[layer][i]))) ) * 2 - 1.0) * neurons_in[layer][i]
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var i : int = 0; while i < structure[layer]:
						var expx : float = pow(2.7182, neurons_in[layer][i])
						var denom = expx * (expx + 2.0) + 2.0; denom *= denom
						layer_der[i] = (expx * ( 4 * (neurons_in[layer][i] + 1.0) + expx * (expx * (expx + 4.0) + 4.0 * neurons_in[layer][i] + 6.0 ))) / denom
						i += 1
				bias_consts[pos] = 1.04904
			BaseNNET.ActivationFunctions.swish:
				callable[pos] = func(layer : int) -> void:
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = neurons_in[layer][i] * (1.0 / ( 1.0 + pow(2.7182, -neurons_in[layer][i])))
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var i : int = 0; while i < structure[layer]:
						var expx : float = pow(2.7182, neurons_in[layer][i])
						var denom = expx + 1.0; denom *= denom
						layer_der[i] = (expx * (neurons_in[layer][i] + expx + 1.0)) / denom
						i += 1
				bias_consts[pos] = 0.927667
			BaseNNET.ActivationFunctions.softmax:
				callable[pos] = func(layer : int) -> void:
					var sum : float = 0.0
					var i : int = 0; while i < structure[layer]:
						neurons_out[layer][i] = pow(2.7182, neurons_in[layer][i])
						sum += neurons_out[layer][i]
						i += 1
					i = 0; while i < structure[layer]:
						neurons_out[layer][i] /= sum
						i += 1
				dcallable[pos] = func(layer : int) -> void:
					layer_der.resize(structure[layer])
					var sum : float = neurons_in[layer][0] / neurons_out[layer][0]
					var i : int = 0; while i < structure[layer]:
						var new_exp = pow(2.7182, neurons_in[layer][i] + BaseNNET.apzero)
						layer_der[i] = ( neurons_out[layer][i] - (new_exp / (sum - neurons_out[layer][i] + new_exp ) ) ) / BaseNNET.apzero
						i += 1
					bias_consts[pos] = 0.5 # according to real calculations it must be 0.0, but I don't want softmax to loose ability on correcting biases when it being applied not on the last layer
	else:
		push_error("Function must be Callable or belong to BaseNNET.ActivationFunctions")

func set_loss_function(function : Variant) -> void:
	if function is Callable:
		loss = function
	elif function is BaseNNET.LossFunctions:
		match function:
			BaseNNET.LossFunctions.MSE:
				loss = func(outputs, targets) -> float:
					var sum : float = 0
					var i : int = 0; while i < outputs.size():
						sum += (targets[i] - outputs[i]) * (targets[i] - outputs[i]); i += 1
					sum /= outputs.size(); return sum
			BaseNNET.LossFunctions.MAE:
				loss = func(outputs, targets) -> float:
					var sum : float = 0
					var i : int = 0; while i < outputs.size():
						var value = targets[i] - outputs[i]; i += 1
						sum += int(value >= 0) * value + int(value < 0) * -value
					sum /= outputs.size(); return sum
			BaseNNET.LossFunctions.BCE:
				loss = func(outputs, targets) -> float:
					var sum : float = 0
					var i : int = 0; while i < outputs.size():
						sum += targets[i] * log(outputs[i]) + (1 - targets[i]) * log(1 - outputs[i]); i += 1
					sum /= outputs.size(); return -sum
			BaseNNET.LossFunctions.CCE:
				loss = func(outputs, targets) -> float:
					var sum : float = 0
					var i : int = 0; while i < outputs.size():
						sum += targets[i] * log(outputs[i]); i += 1
					return -sum
			BaseNNET.LossFunctions.Hinge_loss:
				loss = func(outputs, targets) -> float:
					var sum : float = 0
					var i : int = 0; while i < outputs.size():
						var value = 1 - targets[i] * outputs[i]; i += 1
						sum += int(value > 0) * value
					return sum
			BaseNNET.LossFunctions.Cosine_similarity_loss:
				loss = func(outputs, targets) -> float:
					return 1.0 - dot(normalise(outputs), normalise(targets))
			BaseNNET.LossFunctions.LogCosh_loss:
				loss = func(outputs, targets) -> float:
					var sum : float = 0
					var i : int = 0; while i < outputs.size():
						sum += log(cosh(outputs[i] - targets[i])); i += 1
					return sum
	else:
		push_error("Function must be Callable or belong to BaseNNET.LossFunctions")

func run() -> void:
	var i : int = 1
	while i < structure.size():
		var j : int = 0
		while j < structure[i]:
			neurons_in[i][j] = get_bias.call(i,j)
			var previous_layer_ratio = layer_ratio.call(i - 1)
			var k : int = 0
			while k < structure[i - 1]:
				neurons_in[i][j] += neurons_out[i - 1][k] * weights[i - 1][k][j] * active_neurons[i - 1][k] * previous_layer_ratio
				k += 1
			j += 1
		f[i].call(i)
		i += 1

func set_input(input : Array) -> void:
	if not is_array_valid(input, structure[0]):
		if input.size() == structure[0]:
			push_error("All elements of the input array must be floats")
		else:
			push_error("The quantity of elements in the input array must match number of input neurons. ", "input array size is ", input.size(), " but must be ", structure[0] )
	neurons_out[0] = input.duplicate()

func print_output() -> void:
	print(get_output())

func get_output() -> Array:
	return neurons_out[structure.size() - 1].duplicate()

func use_backpropogation(learning_rate : float) -> void:
	if learning_rate < 0: push_warning("Learning rate is negative! abs function will be automatically applied")
	lr = absf(learning_rate)
	kill_resilient_propagation()
	kill_adam()
	algorithm = BaseNNET.Algorithm.backpropogation

func use_resilient_propagation(new_update_value : float = 0.0125, new_multiplication_factor : float = 1.2, new_reduction_factor : float = 0.3, new_max_step : float = 1.0, new_min_step = 0.000001) -> void:
	if new_update_value < 0: push_warning("Update value is negative! abs function will be automatically applied")
	if new_multiplication_factor < 0: push_warning("Multiplication factor is negative! abs function will be automatically applied")
	if new_reduction_factor < 0: push_warning("Reduction factor is negative! abs function will be automatically applied")
	if new_min_step < 0: push_warning("Min step is negative! abs function will be automatically applied")
	if new_max_step < 0: push_warning("Max step is negative! abs function will be automatically applied")
	update_value = absf(new_update_value)
	multiplication_factor = absf(new_multiplication_factor)
	reduction_factor = absf(new_reduction_factor)
	min_step = absf(new_min_step)
	max_step = absf(new_max_step)
	kill_adam()
	init_setup_resilient_propagation()
	allocate_weights_like_structure(signs)
	allocate_biases_like_structure(bias_signs)
	algorithm = BaseNNET.Algorithm.resilient_propagation

func use_Adam(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, new_weight_decay : float = 0.0) -> void:
	lr = learning_rate
	beta1 = beta_1
	beta2 = beta_2
	weight_decay = new_weight_decay
	allocate_moments()
	kill_resilient_propagation()
	algorithm = BaseNNET.Algorithm.adam

func kill_adam() -> void:
	moments.clear()
	smoments.clear()
	bias_moments.clear()
	bias_moments.clear()

func kill_resilient_propagation() -> void:
	signs.clear()
	bias_signs.clear()

func allocate_moments() -> void:
	allocate_weights_like_structure(moments)
	allocate_weights_like_structure(smoments)
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < structure[i]:
			var k : int = 0
			while k < structure[i + 1]:
				moments[i][j][k] = 0.0
				smoments[i][j][k] = 0.0
				k += 1
			j += 1
		i += 1
	if bias:
		allocate_biases_like_structure(bias_moments)
		allocate_biases_like_structure(bias_smoments)
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				bias_moments[i][j] = 0.0
				bias_smoments[i][j] = 0.0
				j += 1
			i += 1

func train() -> void:
	if algorithm == BaseNNET.Algorithm.backpropogation:
		dropout_neurons.call()
		run() 
		find_deltas()
		correct_weights_backpropagation()
		correct_biases_backpropagation()
	elif algorithm == BaseNNET.Algorithm.resilient_propagation:
		setup_resilient_propagation.call()
		dropout_neurons.call()
		run()
		find_deltas()
		update_signs()
		correct_weights_resilient_propagation()
		correct_biases_resilient_propagation()
	elif algorithm == BaseNNET.Algorithm.adam:
		dropout_neurons.call()
		run()
		find_deltas()
		update_moments()
		correct_weights_adam()
		correct_biases_adam()
	else:
		push_error("Algorithm doesn't exist")

func find_deltas() -> void:
	var error : float = loss.call(neurons_out[structure.size() - 1].duplicate(), target.duplicate())
	df[f.size() - 1].call(f.size() - 1)
	var neuron : int = 0
	while neuron < structure[structure.size() - 1]:
		deltas[deltas.size() - 1][neuron] = neurons_out[neurons_out.size() - 1][neuron] - target[neuron]
		deltas[deltas.size() - 1][neuron] *= layer_der[neuron] * error
		neuron += 1
	var layer : int = deltas.size() - 2
	while layer >= 0:
		df[f.size() - 1].call(layer + 1)
		neuron = 0
		while neuron < structure[layer + 1]:
			deltas[layer][neuron] = 0.0
			var forward_neuron : int = 0
			while forward_neuron < structure[layer + 2]:
				deltas[layer][neuron] += deltas[layer + 1][forward_neuron] * weights[layer + 1][neuron][forward_neuron] * active_neurons[layer + 2][forward_neuron]
				forward_neuron += 1
			deltas[layer][neuron] *= layer_der[neuron] * error * active_neurons[layer + 1][neuron] + (1 - active_neurons[layer + 1][neuron]) * deltas[layer][neuron]
			neuron += 1
		layer -= 1

func correct_weights_backpropagation() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < structure[i]:
			var k : int = 0
			while k < structure[i + 1]:
				weights[i][j][k] -= lr * neurons_out[i][j] * deltas[i][k] * active_neurons[i][j] * active_neurons[i + 1][k]
				k += 1
			j += 1
		i += 1

func correct_weights_resilient_propagation() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < structure[i]:
			var k : int = 0
			while k < structure[i + 1]:
				weights[i][j][k] -= signs[i][j][k] * active_neurons[i][j] * active_neurons[i + 1][k]
				k += 1
			j += 1
		i += 1

func correct_weights_adam() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < structure[i]:
			var k : int = 0
			while k < structure[i + 1]:
				weights[i][j][k] -= lr * ( (moments[i][j][k] / (1.0 - beta1)) / ( sqrt(smoments[i][j][k] / (1.0 - beta2)) + BaseNNET.apzero ) )
				k += 1
			j += 1
		i += 1

func correct_biases_backpropagation() -> void:
	if bias:
		var i : int = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				biases[i][j] -= lr * deltas[i][j] * bias_consts[i] * active_neurons[i + 1][j]
				j += 1
			i += 1

func correct_biases_resilient_propagation() -> void:
	if bias:
		var i : int = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				biases[i][j] -= bias_signs[i][j] * active_neurons[i + 1][j]
				j += 1
			i += 1

func correct_biases_adam() -> void:
	if bias:
		var i : int = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				biases[i][j] -= lr * ( (bias_moments[i][j] / (1.0 - beta1)) / ( sqrt(bias_smoments[i][j] / (1.0 - beta2)) + BaseNNET.apzero ) )
				j += 1
			i += 1

func set_target(desired_output : Array) -> void:
	if not is_array_valid(desired_output, structure[structure.size() - 1]):
		if desired_output.size() == structure[structure.size() - 1]:
			push_error("All elements of the target array must be floats")
		else:
			push_error("The quantity of elements in the target array must match number of output neurons. ", "target array size is ", desired_output.size(), " but must be ", structure[structure.size() - 1] )
	target = desired_output.duplicate()

func update_moments() -> void:
	var i : int = 0; while i < weights.size():
		var j : int = 0; while j < weights[i].size():
			var k : int = 0; while k < weights[i][j].size():
				var delta : float = (neurons_out[i][j] * deltas[i][k] + weight_decay) * active_neurons[i][j] * active_neurons[i + 1][k]
				moments[i][j][k] = beta1 * moments[i][j][k] + (1.0 - beta1) * delta
				smoments[i][j][k] = beta2 * smoments[i][j][k] + (1.0 - beta2) * delta * delta
				k += 1
			j += 1
		i += 1
	if bias:
		i = 0
		while i < bias_signs.size():
			var j : int = 0
			while j < bias_signs[i].size() and active_neurons[i + 1][j]:
				var delta : float = (deltas[i][j] * bias_consts[i] + weight_decay) * active_neurons[i + 1][j]
				bias_moments[i][j] = beta1 * bias_moments[i][j] + (1.0 - beta1) * delta
				bias_smoments[i][j] = beta2 * bias_smoments[i][j] + (1.0 - beta2) * delta * delta
				j += 1
			i += 1

func fill_signs() -> void:
	var i : int = 0
	while i < signs.size():
		var j : int = 0
		while j < structure[i]:
			var k : int = 0
			while k < structure[i + 1]:
				var sign = int(deltas[i][k] * neurons_out[i][j] > 0)
				sign = (1 - sign) * -1 + sign
				signs[i][j][k] = sign * update_value
				k += 1
			j += 1
		i += 1
	if bias:
		i = 0
		while i < bias_signs.size():
			var j : int = 0
			while j < bias_signs[i].size():
				var sign = int(deltas[i][j] * bias_consts[i] > 0)
				sign = (1 - sign) * -1 + sign
				bias_signs[i][j] = sign * update_value
				j += 1
			i += 1

func update_signs() -> void:
	var i : int = 0
	while i < signs.size():
		var j : int = 0
		while j < structure[i]:
			var k : int = 0
			while k < structure[i + 1] and active_neurons[i][j] and active_neurons[i + 1][k]:
				var condition : int = int(signs[i][j][k] > min_step or -signs[i][j][k] > min_step)
				var new_sign = int(deltas[i][k] * neurons_out[i][j] > 0); new_sign = (1 - new_sign) * -1 + new_sign
				var sign_changed : int = int( (deltas[i][k] * neurons_out[i][j] > 0) != (signs[i][j][k] > 0) )
				signs[i][j][k] = signs[i][j][k] * multiplication_factor * (1 - sign_changed) + sign_changed * signs[i][j][k] * reduction_factor
				signs[i][j][k] = -signs[i][j][k] * sign_changed + signs[i][j][k] * (1 - sign_changed)
				signs[i][j][k] = signs[i][j][k] * condition + (1 - condition) * min_step * new_sign
				condition = int(signs[i][j][k] < max_step and -signs[i][j][k] < max_step)
				signs[i][j][k] = signs[i][j][k] * condition + (1 - condition) * max_step * new_sign
				k += 1
			j += 1
		i += 1
	if bias:
		i = 0
		while i < bias_signs.size():
			var j : int = 0
			while j < bias_signs[i].size() and active_neurons[i + 1][j]:
				var condition : int = int(bias_signs[i][j] > min_step or -bias_signs[i][j] > min_step)
				var new_sign = int(deltas[i][j] * bias_consts[i] > 0); new_sign = (1 - new_sign) * -1 + new_sign
				var sign_changed : int = int( (deltas[i][j] * df[0].call(1.0) > 0) != (bias_signs[i][j] > 0) )
				bias_signs[i][j] = bias_signs[i][j] * multiplication_factor * (1 - sign_changed) + sign_changed * bias_signs[i][j] * reduction_factor
				bias_signs[i][j] = -bias_signs[i][j] * sign_changed + bias_signs[i][j] * (1 - sign_changed)
				bias_signs[i][j] = bias_signs[i][j] * condition + (1 - condition) * min_step * new_sign
				condition = int(bias_signs[i][j] < max_step and -bias_signs[i][j] < max_step)
				bias_signs[i][j] = bias_signs[i][j] * condition + (1 - condition) * max_step * new_sign
				j += 1
			i += 1

func get_logits() -> Array:
	return neurons_in[neurons_in.size() - 1].duplicate()

func init_functions(function : BaseNNET.ActivationFunctions) -> void:
	var i : int = 0; while i < structure.size():
		set_function(function, i)
		i += 1

#region Save

func save_data(path : String, binary : bool = true) -> void:
	if not binary:
		nonbinary_save(full_path(path))
	else: binary_save(full_path(path))

func binary_save(path : String) -> void:
	var file := FileAccess.open(path, FileAccess.WRITE)
	if not file.is_open():
		push_error("Couldn't open file: " + error_string(file.get_error())); return
	file.store_string("NNETB ") # NNET Binary
	file.store_string(version + "\n")
	file.store_16(structure.size())
	file.store_8(int(bias))
	for layer in structure: file.store_32(layer)
	# saving weights
	var i : int = 0; while i < weights.size():
		var j : int = 0; while j < weights[i].size():
			var k : int = 0; while k < weights[i][j].size():
				file.store_double(weights[i][j][k])
				k += 1
			j += 1
		i += 1
	# saving biases
	i = 0; while i < biases.size():
		var j : int = 0; while j < biases[i].size():
			file.store_double(biases[i][j])
			j += 1
		i += 1
	file.close()

func nonbinary_save(path : String) -> void:
	var file := FileAccess.open(path, FileAccess.WRITE)
	if not file.is_open():
		push_error("Couldn't open file: " + error_string(file.get_error())); return
	file.store_string("NNETN ") # NNET Non-binary
	file.store_string(version + "\n")
	file.store_string("structure : " + str(structure).erase(0,1).left(-1).replace(",","") + "\n")
	file.store_string("weights : " + str(total_weights()) + "\n")
	file.store_string("biases : " + str(total_biases()) + "\n")
	# saving weights
	var i : int = 0; while i < weights.size():
		var j : int = 0; while j < weights[i].size():
			var k : int = 0; while k < weights[i][j].size():
				file.store_string("weight : " + str(weights[i][j][k]) + "\n")
				k += 1
			j += 1
		i += 1
	#saving biases
	i = 0; while i < biases.size():
		var j : int = 0; while j < biases[i].size():
			file.store_string("bias : " + str(biases[i][j]) + "\n")
			j += 1
		i += 1
	file.close()

func total_biases() -> int:
	var sum : int = 0
	var i : int = 0; while i < biases.size():
		sum += biases[i].size()
		i += 1
	return sum

func total_weights() -> int:
	var sum : int = 0
	var i : int = 0; while i < weights.size():
		sum += structure[i] * structure[i + 1]
		i += 1
	return sum

#endregion

#region Load

func load_data(path : String) -> void:
	var file := FileAccess.open(full_path(path), FileAccess.READ)
	if not file.is_open():
		push_error("Couldn't open file: " + error_string(file.get_error())); return
	var string : String = file.get_buffer(5).get_string_from_utf8(); file.close()
	if string.begins_with("NNETB"):
		binary_load(full_path(path))
	elif string.begins_with("NNETN"):
		nonbinary_load(full_path(path))
	else:
		push_warning("File is corrupted. Unknown pattern")

func binary_load(path : String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)
	if not file.is_open():
		push_error("Couldn't open file: " + error_string(file.get_error())); return
	file.get_buffer(12); var structure_size : int = file.get_16();
	var new_structure : Array = []; var use_bias : bool = bool(file.get_8())
	var i : int = 0; while i < structure_size:
		new_structure.append(file.get_32()); i += 1
	_init(new_structure, 0.1, use_bias)
	# loading weights
	i = 0; while i < weights.size():
		var j : int = 0; while j < weights[i].size():
			var k : int = 0; while k < weights[i][j].size():
				weights[i][j][k] = file.get_double()
				k += 1
			j += 1
		i += 1
	# loading biases
	i = 0; while i < biases.size():
		var j : int = 0; while j < biases[i].size():
			biases[i][j] = file.get_double()
			j += 1
		i += 1
	file.close()

func nonbinary_load(path : String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)
	if not file.is_open():
		push_error("Couldn't open file: " + error_string(file.get_error())); return
	file.get_buffer(12)
	var new_structure : Array = Array(file.get_line().get_slice(" : ", 1).split(" ")).map(func(string): return string.to_int())
	file.get_line(); var use_bias : bool = file.get_line().get_slice(" : ", 1).to_int() != 0
	_init(new_structure, 0.1, use_bias)
	# loading weights
	var i : int = 0; while i < weights.size():
		var j : int = 0; while j < weights[i].size():
			var k : int = 0; while k < weights[i][j].size():
				weights[i][j][k] = file.get_line().get_slice(" : ", 1).to_float()
				k += 1
			j += 1
		i += 1
	# loading biases
	i = 0; while i < biases.size():
		var j : int = 0; while j < biases[i].size():
			biases[i][j] = file.get_line().get_slice(" : ", 1).to_float()
			j += 1
		i += 1
	file.close()

#endregion

func assign(neural_network : NNET) -> void:
	pass

func duplicate() -> NNET:
	var neural_network : NNET = NNET.new([1,1],0.0,false)
	neural_network.assign(self)
	return neural_network

func dot(vec1 : Array, vec2 : Array) -> float:
	var sum : float = 0
	var i : int = 0
	while i < vec1.size():
		sum += vec1[i] * vec2[i]
		i += 1
	return sum

func length(vec : Array) -> float:
	var sum : float = 0
	var i : int = 0
	while i < vec.size():
		sum += vec[i] * vec[i]
		i += 1
	return sqrt(sum)

func normalise(vec : Array) -> Array:
	var dupl_vec : Array = vec.duplicate()
	var vec_length = length(vec)
	var i : int = 0; while i < vec.size():
		dupl_vec[i] /= vec_length
		i += 1
	return dupl_vec

func enable_dropout(probability : float) -> void:
	if probability < 0.0:
		push_warning("probability is lesser than zero. Dropout won't be used"); return
	elif probability > 1.0:
		push_warning("probability is greater than one. Dropout won't be used"); return
	p = probability
	layer_ratio = func(layer : int) -> float:
		var active : int = 0
		var i : int = 0; while i < active_neurons[layer].size():
			active += active_neurons[layer][i]
			i += 1
		return float(active_neurons[layer].size()) / active
	dropout_neurons = func() -> void:
		var i : int = 1; while i < structure.size() - 1:
			var j : int = 0; while j < structure[i]:
				active_neurons[i][j] = int(randi() % 1000000 > int(p * 1000000))
				j += 1
			active_neurons[i][randi() % structure[i]] = 1 # making sure at least one neuron in the layer is active
			i += 1

func disable_dropout() -> void:
	layer_ratio = func(_layer : int) -> float : return 1.0
	dropout_neurons = func() -> void: pass
	make_all_neurons_active()

func make_all_neurons_active() -> void:
	var i : int = 0; while i < structure.size():
		var j : int = 0; while j < active_neurons[i].size():
			active_neurons[i][j] = 1
			j += 1
		i += 1

func init_active_neurons() -> void:
	allocate_neurons_like_structure(active_neurons)
