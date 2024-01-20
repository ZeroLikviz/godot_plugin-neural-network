## Class for basic neural networks
## 
## provides basic functional for controlling neural network [br]
## usually you need 5 functions:
## - [method NNET.set_input] to set input
## - [method NNET.set_desired_output] to set desired output
## - [method NNET.run] to run neural_network
## - [method NNET.get_output] to get output from neural_network after you ran it (otherwise output will contain only zeros)
## - [method NNET.train] to train neural_network

@icon("res://addons/neural_network/NNET icon.png")
class_name NNET
## R_0_1 from 0 to 1 [br]
## R_M1_1 from -1 to 1
enum RangeN {
	R_0_1,
	R_M1_1
}

## output of neurons
var neurons_out: Array[Array] = []
## input of neurons
var neurons_in: Array[Array] = []
## data used in train function
var deltas: Array[Array] = []
## biases
var biases: Array[Array] = []
## this member contains weights
var weights: Array = []
## desired output
var output: Array = []
## construction of a neural network, like [1,8,2] [br]
## 1 input neuron [br]
## 1 hidden layer, which contains 8 neurons [br]
## 2 output neurons [br]
## [br]
## "layers" contains amount of neurons in each layer, it doesn't contain layers itself
var layers: Array = []
## it is a learning rate. Opting for a higher learning rate facilitates accelerated learning; however, it may come at the expense of compromising the quality of the obtained outcomes. 
var learning_rate: float
## amount of layers
var layers_size: int
## index of the last layer
var last_layer: int
## it is a component that indicates when a neural network should or should not use biases (offsets)
var is_using_bias: bool = false
## it is range, it can be only from 0 to 1 (RangeN.R_0_1), or from -1 to 1 (RangeN.R_M1_1)
var range_member: RangeN
## The variable tfd, short for "true f'()", is utilized within the "train" function. When tfd is set to false, the function f'() undergoes substitution with the value of 1.0.
var tfd: bool
## This variable used for optimization. It helps to avoid unnecessary runs.
var train_run: bool = true

func _init(layers_construction: Array = [1,1], learning_rate_a: float = 1.0, use_bias: bool = true, range_a: RangeN = RangeN.R_0_1, tfd_a : bool = false) -> void:
	learning_rate = learning_rate_a
	is_using_bias = use_bias
	range_member = range_a
	tfd = tfd_a
	var minw: float = -1.0
	var maxw: float = 1.0
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
		neurons_out[i].resize(layers[i])
		deltas[i].resize(layers[i])
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
		weights[i] = randf_range(minw, maxw)
		i += 1
	output.resize(layers[last_layer])
	output.fill(0.0)

	if is_using_bias:
		biases.resize(layers.size() - 1)
		for el in biases:
			el = []
		i = 0
		while i < biases.size():
			biases[i].resize(layers[i + 1])
			i += 1
			var j : int = 0
			while j < biases[i].size():
				biases[i][j] = randf_range(minw, maxw)
				j += 1

## This function is responsible for assigning the desired output value for the neural network.
func set_desired_output(desired_output: Array[float]) -> void:
	assert(desired_output.size() == layers[last_layer], "set_desired_output: sizes doesn't match")
	output = desired_output.duplicate(true)
	train_run = true

## This function is responsible for assigning the input value for the neural network.
func set_input(input: Array[float]) -> void:
	assert(input.size() == layers[0], "set_input: sizes doesn't match")
	neurons_in[0] = input.duplicate(true)
	neurons_out[0] = input.duplicate(true)
	train_run = true

## you shouldn't use this function, but in case if you want here how it works: [br]
## provide two neurons present in distinct layers, and the function shall yield the weight index connecting these two neurons.
## here's an example:
## [codeblock]
##     ...
##     var index : int = neural_network.get_weight(0,0,1,4)
##     #don't do what I'm gonna show you
##     neural_network.weights[index] = SomeFloatValueThatEvilUserUsedToChangeOneOfTheWeights
## [/codeblock]
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
## It is a sigmoid function when range is from 0 to 1, and It id a tanh function when range is from -1 to 1
func f(x: float) -> float:
	if range_member== RangeN.R_M1_1:
		return (2.0 / (1.0 + pow(2.7182, -2.0 * x))) - 1.0
	return 1.0 / (pow(2.7182, -x) + 1.0)
## f'(x)
func fd(x: float) -> float:
	if range_member== RangeN.R_M1_1:
		return 1.0 - pow(f(x), 2) * int(tfd)
	return (f(x) * (1.0 - f(x)) * int(tfd)) + (1 - int(tfd))
## This function employs a neural network model to execute computations and generate output
func run() -> void:
	train_run = false
	var layer : int = 1
	while layer < layers_size:
		var neuron : int = 0
		while neuron < layers[layer]:
			neurons_in[layer][neuron] = biases[layer - 1][neuron] * int(is_using_bias)
			var back_neuron : int = 0
			while back_neuron < layers[layer - 1]:
				neurons_in[layer][neuron] += weights[get_weight(layer, neuron, layer - 1, back_neuron)] * neurons_out[layer - 1][back_neuron]
				back_neuron += 1
			neurons_out[layer][neuron] = f(neurons_in[layer][neuron])
			neuron += 1
		layer += 1

func compute_deltas() -> void:
	var neuron : int = 0
	while neuron < layers[last_layer]:
		deltas[last_layer][neuron] = neurons_out[last_layer][neuron] - output[neuron]
		deltas[last_layer][neuron] *= fd(neurons_in[last_layer][neuron])
		neuron += 1
	
	var layer : int = last_layer - 1
	while layer > 0:
		neuron = 0
		while neuron < layers[layer]:
			deltas[layer][neuron] = 0.0
			var after_neuron : int = 0
			while after_neuron < layers[layer + 1]:
				deltas[layer][neuron] += deltas[layer + 1][after_neuron] * weights[get_weight(layer, neuron, layer + 1, after_neuron)]
				after_neuron += 1
			deltas[layer][neuron] *= fd(neurons_in[layer][neuron])
			neuron += 1
		layer -= 1
## Member laps refers to the number of iterations through which this function will be repeatedly executed [br]
##
## [codeblock]neural_network.train(someValue)[/codeblock]
## equals to
## [codeblock]
## var i : int = 0
## while i < someValue:
##     i += 1
##     neural_network.train()
## [/codeblock]
## or
## [codeblock]
## for i in range(someValue):
##     neural_network.train()
## [/codeblock]
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
					weights[get_weight(layer, neuron, layer + 1, after_neuron)] -= learning_rate * deltas[layer + 1][after_neuron] * neurons_out[layer][neuron]
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

## returns a copy of the last layer of neurons_out ( see [member NNET.neurons_out] )
func get_output() -> Array:
	return neurons_out[last_layer].duplicate(true)

## prints result (output) in console
func show_result() -> void:
	var size: int = layers[last_layer]
	var console : String = ""
	var i : int = 0
	while i < size:
		console += "[" + str(neurons_out[last_layer][i]) + "]" + " "
		i += 1
	print(console)


func duplicate():
	var buffer = NNET.new(layers, learning_rate, is_using_bias, range_member, tfd)
	buffer.output.assign(output)
	buffer.neurons_out[last_layer].assign(neurons_out[last_layer])
	buffer.neurons_out[0].assign(neurons_out[0])
	buffer.neurons_in[0].assign(neurons_in[0])
	buffer.weights.assign(weights)
	buffer.biases.assign(biases.duplicate(true))
	return buffer

func assign(buffer : NNET) -> void:
	output.assign(buffer.output)
	layers.assign(buffer.layers)
	neurons_in.assign(buffer.neurons_in.duplicate(true))
	neurons_out.assign(buffer.neurons_out.duplicate(true))
	biases.assign(buffer.biases.duplicate(true))
	weights.assign(buffer.weights)
	deltas.assign(buffer.deltas.duplicate(true))
	tfd = buffer.tfd
	train_run = buffer.train_run
	range_member = buffer.range_member
	learning_rate = buffer.learning_rate
	is_using_bias = buffer.is_using_bias
	layers_size = buffer.layers_size
	last_layer = buffer.last_layer

##  accepts 1 parameter containing the file name with the extension. If the parameter contains the full path, the file will be located at the specified path
func save_data(file_name : String) -> void:
	if not DirAccess.dir_exists_absolute("res://addons/neural_network/data/"):
		DirAccess.make_dir_absolute("res://addons/neural_network/data")
	var file =  FileAccess.open("res://addons/neural_network/data/" + file_name, FileAccess.WRITE)
	if file_name.begins_with("res://") or file_name.begins_with("user://"):
		file.close()
		file = FileAccess.open(file_name, FileAccess.WRITE)
	for layer in layers:
		file.store_64(layer)
	for weight in weights:
		file.store_double(weight)
	for bias_layer in biases:
		for bias in bias_layer:
			file.store_double(bias)
	file.close()

## accepts 1 parameter containing the file name with the extension. If the parameter contains the full path, the data will be loaded from the full path
func load_data(file_name : String) -> void:
	var buffer : NNET = duplicate()
	var corrupted: bool = false
	
	var file =  FileAccess.open("res://addons/neural_network/data/" + file_name, FileAccess.READ)
	if file_name.begins_with("res://") or file_name.begins_with("user://"):
		file.close()
		file = FileAccess.open(file_name, FileAccess.READ)
	for layer in layers:
		assert(layer == file.get_64(), "neural network structure doesn't match")
	var i : int = 0
	while i < weights.size():
		var boolean : bool = not file.eof_reached()
		if not boolean: corrupted = true
		assert(boolean, "neural network structure doesn't match")
		
		weights[i] = file.get_double()
		i += 1
	i = 0
	while i < biases.size():
		var j : int = 0
		while j < biases[i].size():
			var boolean : bool = not file.eof_reached()
			if not boolean: corrupted = true
			assert(boolean, "neural network structure doesn't match")
			biases[i][j] = file.get_double()
			j += 1
		i += 1
	
	file.get_double()
	assert(file.eof_reached(),"neural network structure doesn't match")
	if corrupted or not file.eof_reached():
		assign(buffer)
	file.close()
	
