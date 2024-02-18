@icon("res://addons/neural_network/NNET icon.png")
class_name NNET

var get_bias : Callable = func(layer : int, neuron : int) -> float:
	return 0.0
var f : Callable
var fd : Callable = func(x : float) -> float:
	return 1.0
var neurons_out: Array[Array] = []
var neurons_in: Array[Array] = []
var deltas: Array[Array] = []
var biases: Array[Array] = []
var weights: Array = []
var output: Array = []
var layers: Array = []
var learning_rate: float
var layers_size: int
var last_layer: int
var is_using_bias: bool = false
var true_fd: bool
var train_run: bool = true
var weights_table : Array = []

enum ActivationFunction
{
	linear            ,
	sigmoid           ,
	logistic = sigmoid,
	ReLU              ,
	custom
}

var activation_function_info : int = ActivationFunction.sigmoid

class TheRange:
	var min_value : float
	var max_value : float
	
	# provide it parameter that ranges from 0 to 1
	func rearrange(value : float) -> float:
		return min_value + value * (max_value - min_value)
	
	func restore(value : float) -> float:
		return (value - min_value) / (max_value - min_value)
	
	func _init(min : float, max : float) -> void:
		var boolean : int = int(min > max)
		min_value = min * (1 - boolean) + max * boolean
		max_value = max * (1 - boolean) + min * boolean
	
	func duplicate() -> TheRange:
		return TheRange.new(min_value, max_value)
	func assign(buffer : TheRange) -> void:
		min_value = buffer.min_value
		max_value = buffer.max_value

func _init(layers_construction: Array = [1,1], learning_rate_value: float = 1.0, use_bias: bool = true, true_fd_value : bool = true) -> void:
	learning_rate = learning_rate_value
	is_using_bias = use_bias
	
	set_function(ActivationFunction.sigmoid)
	
	if true_fd_value:
		fd = func (x : float) -> float:
			return 1.0 - pow(f.call(x), 2)
	true_fd = true_fd_value
	
	layers = layers_construction
	layers_size = layers.size()
	last_layer = layers_size - 1
	assert(layers_size > 0, "NNET: layers_size less or equals to 0")
	
	var i : int = 0
	while i < layers_size:
		assert(layers[i] > 0, "NNET: any layer cant have size 0 or less")
		neurons_out.append([])
		neurons_in.append([])
		deltas.append([])
		
		neurons_in[i].resize(layers[i])
		neurons_in[i].fill(0.0)
		neurons_out[i].resize(layers[i])
		neurons_out[i].fill(0.0)
		deltas[i].resize(layers[i])
		deltas[i].fill(0.0)
		i += 1
	i = 0
	
	deltas[0].resize(0)
	var weights_size: int = 0
	while i < layers_size - 1:
		weights_size += layers[i] * layers[i + 1]
		i += 1
	weights.resize(weights_size)
	i = 0
	
	
	while i < weights_size:
		weights[i] = randf_range(-1.0, 1.0)
		i += 1
	output.resize(layers[last_layer])
	output.fill(0.0)

	if is_using_bias:
		get_bias = func(layer : int, neuron : int) -> float:
			return biases[layer][neuron]
		biases.resize(layers.size() - 1)
		for el in biases:
			el = []
		i = 0
		while i < biases.size():
			biases[i].resize(layers[i + 1])
			var j : int = 0
			while j < biases[i].size():
				biases[i][j] = randf_range(-1.0, 1.0)
				j += 1
			i += 1
	
	fill_table_of_weights()

func fill_table_of_weights() -> void:
	weights_table.resize(layers_size - 1)
	var i : int = 0
	while i < layers_size - 1:
		weights_table[i] = []
		weights_table[i].resize(layers[i])
		var j : int = 0
		while j < layers[i]:
			weights_table[i][j] = []
			weights_table[i][j].resize(layers[i + 1])
			var z : int = 0
			while z < layers[i + 1]:
				weights_table[i][j][z] = get_weight(i, j, i + 1, z)
				z += 1
			j += 1
		i += 1

func get_tabled_weight(layer1: int, neuron1: int, layer2: int, neuron2: int) -> int:
	var buffer = neuron1
	var bool1 = int(layer1 > layer2)
	var bool2 = 1 - bool1
	layer1 = min(layer1, layer2)
	neuron1 = neuron2 * bool1 + neuron1 * bool2
	neuron2 = buffer * bool1 + neuron2 * bool2
	return weights_table[layer1][neuron1][neuron2]

func set_function(function : ActivationFunction) -> void:
	match function:
		ActivationFunction.linear:
			f = func (x : float) -> float:
				return x
			if true_fd:
				fd = func (x : float) -> float:
					return 1.0
		ActivationFunction.sigmoid:
			f = func (x : float) -> float:
				return 1.0 / (pow(2.7182, -x) + 1.0)
			if true_fd:
				fd = func (x : float) -> float:
					return f.call(x) * (1.0 - f.call(x))
		ActivationFunction.ReLU:
			f = func (x : float) -> float:
				return max(0.0, x)
			if true_fd:
				fd = func (x : float) -> float:
					return 1.0;
	activation_function_info = function

func set_custom_function(function : Callable) -> void:
	assert(function.get_bound_arguments_count() == 1, "function must have 1 paremeter")
	assert(typeof(function.get_bound_arguments()[0]) == TYPE_FLOAT, "argument type should be x")
	f = function
	activation_function_info = ActivationFunction.custom
	if true_fd:
		fd = func (x : float) -> float:
			return (f.call(x + 0.00001) - f.call(x)) / 0.00001

func set_desired_output(desired_output: Array[float]) -> void:
	assert(desired_output.size() == layers[last_layer], "set_desired_output: sizes doesn't fit")
	output = desired_output.duplicate(true)
	train_run = true

func set_input(input: Array[float]) -> void:
	assert(input.size() == layers[0], "set_input: sizes doesn't fit")
	neurons_in[0] = input.duplicate(true)
	neurons_out[0] = input.duplicate(true)
	train_run = true

func get_weight(layer1: int, neuron1: int, layer2: int, neuron2: int) -> int:
	if layer1 > layer2:
		var buff1 = layer1
		layer1 = layer2
		layer2 = buff1
		var buff2 = neuron1
		neuron1 = neuron2
		neuron2 = buff2
	var weight_position: int = 0
	var i: int = 0
	while i < layer1:
		weight_position += layers[i] * layers[i + 1]
		i += 1
	weight_position += neuron1 * layers[layer2] + neuron2
	return weight_position

func run() -> void:
	train_run = false
	var layer : int = 1
	while layer < layers_size:
		var neuron : int = 0
		while neuron < layers[layer]:
			neurons_in[layer][neuron] = get_bias.call(layer - 1,neuron)
			var back_neuron : int = 0
			while back_neuron < layers[layer - 1]:
				neurons_in[layer][neuron] += weights[get_tabled_weight(layer, neuron, layer - 1, back_neuron)] * neurons_out[layer - 1][back_neuron]
				back_neuron += 1
			neurons_out[layer][neuron] = f.call(neurons_in[layer][neuron])
			neuron += 1
		layer += 1

func compute_deltas() -> void:
	var neuron : int = 0
	while neuron < layers[last_layer]:
		deltas[last_layer][neuron] = neurons_out[last_layer][neuron] - output[neuron]
		deltas[last_layer][neuron] *= fd.call(neurons_in[last_layer][neuron])
		neuron += 1
	
	var layer : int = last_layer - 1
	while layer > 0:
		neuron = 0
		while neuron < layers[layer]:
			deltas[layer][neuron] = 0.0
			var after_neuron : int = 0
			while after_neuron < layers[layer + 1]:
				deltas[layer][neuron] += deltas[layer + 1][after_neuron] * weights[get_tabled_weight(layer, neuron, layer + 1, after_neuron)]
				after_neuron += 1
			deltas[layer][neuron] *= fd.call(neurons_in[layer][neuron])
			neuron += 1
		layer -= 1

func train(laps : int = 1) -> void:
	var lap : int = 0
	while lap < laps:
		lap += 1
		
		if train_run:
			run()
			compute_deltas()
		train_run = true
		
		var layer : int = last_layer - 1
		while layer > -1:
			var neuron : int = 0
			while neuron < layers[layer]:
				var after_neuron : int = 0
				while after_neuron < layers[layer + 1]:
					weights[get_tabled_weight(layer, neuron, layer + 1, after_neuron)] -= learning_rate * deltas[layer + 1][after_neuron] * neurons_out[layer][neuron]
					after_neuron += 1
				neuron += 1
			layer -= 1
		
		if is_using_bias:
			layer = last_layer - 1
			while layer > -1:
				var neuron : int = 0
				while neuron < layers[layer + 1]:
					biases[layer][neuron] -= learning_rate * deltas[layer + 1][neuron]
					neuron += 1
				layer -= 1

func get_output(transform : bool = false) -> Array:
	var buffer = neurons_out[last_layer].duplicate(true)
	var i : int = 0
	if activation_function_info == ActivationFunction.sigmoid:
		while i < buffer.size() * int(transform):
			buffer[i] = buffer[i] * 2 - 1
			i += 1
	return buffer

func print_output(rearrange_by_range : bool = false) -> void:
	print(get_output(rearrange_by_range))

func duplicate():
	var buffer = NNET.new(layers, learning_rate, is_using_bias, true_fd)
	buffer.output.assign(output)
	buffer.neurons_out[last_layer].assign(neurons_out[last_layer])
	buffer.neurons_out[0].assign(neurons_out[0])
	buffer.neurons_in[0].assign(neurons_in[0])
	buffer.weights.assign(weights)
	buffer.biases.assign(biases.duplicate(true))
	if activation_function_info != ActivationFunction.custom:
		buffer.set_function(activation_function_info)
	else: buffer.set_custom_function(f)
	return buffer

func assign(buffer : NNET) -> void:
	if buffer.activation_function_info == ActivationFunction.custom:
		set_custom_function(buffer.f)
	else: set_function(buffer.activation_function_info)
	output.assign(buffer.output)
	layers.assign(buffer.layers)
	neurons_in.assign(buffer.neurons_in.duplicate(true))
	neurons_out.assign(buffer.neurons_out.duplicate(true))
	biases.assign(buffer.biases.duplicate(true))
	weights.assign(buffer.weights)
	deltas.assign(buffer.deltas.duplicate(true))
	true_fd = buffer.true_fd
	train_run = buffer.train_run
	learning_rate = buffer.learning_rate
	is_using_bias = buffer.is_using_bias
	layers_size = buffer.layers_size
	last_layer = buffer.last_layer
	weights_table.resize(buffer.weights_table.size())
	var i : int = 0
	while i < weights_table.size():
		weights_table[i] = buffer.weights_table[i].duplicate(true)
		i += 1

func save_data(file_name : String) -> void:
	if not DirAccess.dir_exists_absolute("res://addons/neural_network/data/"):
		DirAccess.make_dir_absolute("res://addons/neural_network/data")
	var file =  FileAccess.open("res://addons/neural_network/data/" + file_name, FileAccess.WRITE)
	if file_name.begins_with("res://") or file_name.begins_with("user://"):
		file.close()
		file = FileAccess.open(file_name, FileAccess.WRITE)
	file.store_8(int(activation_function_info))
	file.store_8(int(is_using_bias))
	file.store_8(int(true_fd))
	file.store_64(layers_size)
	for layer in layers:
		file.store_64(layer)
	for weight in weights:
		file.store_double(weight)
	for bias_layer in biases:
		for bias in bias_layer:
			file.store_double(bias)
	file.close()

func load_data(file_name : String) -> int:
	var buffer : NNET = duplicate()
	var corrupted: bool = false
	
	var file =  FileAccess.open("res://addons/neural_network/data/" + file_name, FileAccess.READ)
	if file_name.begins_with("res://") or file_name.begins_with("user://"):
		file.close()
		file = FileAccess.open(file_name, FileAccess.READ)
	
	var info : int = file.get_8()
	if file.get_8() != int(is_using_bias):
		printerr("NNET.gd printerr LINE 348: neural network structure doesn't fit")
		assign(buffer)
		return -1
	true_fd = file.get_8(); set_function(info)
	if layers_size != file.get_64():
		printerr("NNET.gd printerr LINE 353: neural network structure doesn't fit")
		assign(buffer)
		return -1
	for layer in layers:
		if layer != file.get_64():
			printerr("NNET.gd printerr LINE 358: neural network structure doesn't fit")
			assign(buffer)
			return -1
	var i : int = 0
	while i < weights.size():
		var boolean : bool = not file.eof_reached()
		if not boolean:
			corrupted = true
			printerr("NNET.gd printerr LINE 366: neural network structure doesn't fit")
			assign(buffer)
			return -1
		weights[i] = file.get_double()
		i += 1
	i = 0
	while i < biases.size():
		var j : int = 0
		while j < biases[i].size():
			var boolean : bool = not file.eof_reached()
			if not boolean:
				corrupted = true
				printerr("NNET.gd printerr LINE 378: neural network structure doesn't fit")
				assign(buffer)
				return -1
			biases[i][j] = file.get_double()
			j += 1
		i += 1
	
	file.get_double()
	if corrupted or not file.eof_reached():
		assign(buffer)
		printerr("NNET.gd printerr LINE 388: neural network structure doesn't fit")
		return -1
	file.close()
	fill_table_of_weights()
	return 0

func copy_from_file(file_name : String) -> void:
	var file = FileAccess.open("res://addons/neural_network/data/" + file_name, FileAccess.READ)
	if file_name.begins_with("res://") or file_name.begins_with("user://"):
		file.close()
		file = FileAccess.open(file_name, FileAccess.READ)
	var function_info_file = file.get_8()
	var bias_fbool : bool = file.get_8()
	var true_fd_fvalue = file.get_8()
	var size = file.get_64()
	var structure_farray = []
	var i : int = 0
	while i < size:
		structure_farray.append(file.get_64())
		i += 1
	var buffer : NNET = NNET.new(structure_farray, learning_rate, bias_fbool, true_fd_fvalue)
	file.close()
	buffer.load_data(file_name)
	assign(buffer)
	set_function(function_info_file)
	fill_table_of_weights()

func print_info(NeuralNetworkName : String, offset : int = 0) -> void:
	var offset_string : String = ""
	var iterator : int = 0
	while iterator < offset:
		offset_string += " "
		iterator += 1
	print_rich(offset_string + "[color=white][u]" + NeuralNetworkName + "[/u][/color]" + " :")
	print(offset_string + "    structure:           " + str(layers))
	print(offset_string + "    bias using:          " + str(is_using_bias))
	print(offset_string + "    learning rate:       " + str(learning_rate))
	print(offset_string + "    true f'():           " + str(true_fd))
	print(offset_string + "    weights:             " + str(weights))
	var function : String = "custom"
	match activation_function_info:
		ActivationFunction.linear  : function = "linear"
		ActivationFunction.sigmoid : function = "sigmoid (logistic)"
		ActivationFunction.ReLU    : function = "ReLU"
	print(offset_string + "    activation function: " + function)
