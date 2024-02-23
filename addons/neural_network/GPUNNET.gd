@icon("res://addons/neural_network/GPUNNET icon.png")
class_name GPUNNET

enum {RUN = 0, FIND_DELTAS = 1, TRAIN = 2, BIAS_TRAIN = 3}

var device                 : RenderingDevice

var common_data            : RID
var bias_storage           : RID
var deltas_storage         : RID
var neurons_storage        : RID
var weights_storage        : RID
var structure_storage      : RID
var desired_output_storage : RID

var pipeline     : Array[RID]
var uniform_set  : Array[RID]
var shader       : Array[RID]

var learning_rate : float
var layers : Array[int]
var last_layer : int
var layers_size : int
var is_bias_used : bool
var neurons_max : int
var x_workgroups : int

func init_snp(index : int, file : String, uniforms : Array[RDUniform]) -> void:
	shader[index] = device.shader_create_from_spirv(load(file).get_spirv())
	pipeline[index] = device.compute_pipeline_create(shader[index])

func _init(layers_construction: Array[int] = [1,1], learning_rate_value: float = 1.0, use_bias: bool = true) -> void:
	assert(layers_construction.size() >= 2, "GPUNNET _init LINE " + str(get_stack()[0]["line"]) + ": Neural network should have at least 2 layers")
	for layer in layers_construction:
		assert(layer >= 1, "GPUNNET _init LINE " + str(get_stack()[0]["line"]) + ": Amount of neurons in layer can't be less than 1")
	
	device = RenderingServer.create_local_rendering_device()
	shader.resize(4)
	pipeline.resize(4)
	uniform_set.resize(4)
	
	is_bias_used = use_bias
	learning_rate = learning_rate_value
	last_layer=layers_construction.size()-1
	layers_size= layers_construction.size()
	layers= layers_construction.duplicate()
	neurons_max = layers_construction.max()
	
	var uint_layers : PackedByteArray = PackedByteArray([])
	uint_layers.resize(layers_size * 4); var i : int = 0;
	while i < layers_size: uint_layers.encode_u32(i * 4, layers[i]); i += 1
	
	var random_weights : PackedFloat32Array = PackedFloat32Array([])
	random_weights.resize(last_layer * neurons_max * neurons_max)
	i = 0; while i < random_weights.size(): random_weights[i] = randf_range(-1.0, 1.0); i += 1
	
	common_data            = device.storage_buffer_create(28)
	bias_storage           = device.storage_buffer_create(last_layer * neurons_max * 4)
	deltas_storage         = device.storage_buffer_create(last_layer * neurons_max * 4)
	weights_storage        = device.storage_buffer_create(last_layer * neurons_max * neurons_max * 4, random_weights.to_byte_array())
	structure_storage      = device.storage_buffer_create(layers_size * 4, uint_layers)
	neurons_storage        = device.storage_buffer_create(layers_size * neurons_max* 4)
	desired_output_storage = device.storage_buffer_create(layers[last_layer] * 4)
	
	fill_data()
	
	var uniforms : Array[RDUniform] = []
	
	uniforms.append(create_uniform(common_data, 0))
	uniforms.append(create_uniform(bias_storage,  1))
	uniforms.append(create_uniform(deltas_storage,  2))
	uniforms.append(create_uniform(weights_storage,   3))
	uniforms.append(create_uniform(structure_storage,   4))
	uniforms.append(create_uniform(neurons_storage,       5))
	uniforms.append(create_uniform(desired_output_storage,  6))
	
	init_snp(RUN        , "res://addons/neural_network/GPUNNET_RUN.glsl"        , uniforms)
	init_snp(FIND_DELTAS, "res://addons/neural_network/GPUNNET_FIND_DELTAS.glsl", uniforms)
	init_snp(TRAIN      , "res://addons/neural_network/GPUNNET_TRAIN.glsl"      , uniforms)
	init_snp(BIAS_TRAIN , "res://addons/neural_network/GPUNNET_BIAS_TRAIN.glsl" , uniforms)
	uniform_set[RUN] = device.uniform_set_create(get_unifroms_array(uniforms, [0,1,3,4,5]), shader[RUN], 0)
	uniform_set[FIND_DELTAS] = device.uniform_set_create(get_unifroms_array(uniforms, [0,2,3,4,5,6]), shader[FIND_DELTAS], 0)
	uniform_set[TRAIN] = device.uniform_set_create(get_unifroms_array(uniforms, [0,2,3,4,5]), shader[TRAIN], 0)
	uniform_set[BIAS_TRAIN] = device.uniform_set_create(get_unifroms_array(uniforms, [0,1,2,4]), shader[BIAS_TRAIN], 0)
	
	x_workgroups = int(floor(float(neurons_max) / 64.0)) + 1

func get_unifroms_array(uniforms : Array[RDUniform], pos : Array[int]) -> Array[RDUniform]:
	var uniforms_array : Array[RDUniform] = []
	uniforms_array.resize(pos.size())
	var i : int = 0
	for unif in pos:
		uniforms_array[i] = uniforms[unif]
		i += 1
	return uniforms_array

func create_uniform(rid : RID, binding : int) -> RDUniform:
	var uniform = RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding = binding
	uniform.add_id(rid)
	return uniform

func submit(mode : int) -> void:
	var compute_list := device.compute_list_begin()
	device.compute_list_bind_compute_pipeline(compute_list, pipeline[mode])
	device.compute_list_bind_uniform_set(compute_list, uniform_set[mode], 0)
	device.compute_list_dispatch(compute_list, x_workgroups,1,1)
	device.compute_list_end()
	
	device.submit()
	device.sync()

func set_layer(layer : int) -> void:
	var data : PackedByteArray = PackedByteArray([]); data.resize(4)
	data.encode_u32(0, layer)
	device.buffer_update(common_data, 16, 4, data)

func fill_data() -> void:
	var data : PackedByteArray = PackedByteArray([])
	data.resize(28)
	data.encode_float(  0, learning_rate)
	data.encode_u32(    4, last_layer   )
	data.encode_u32(    8, layers_size  )
	data.encode_u32(   12, neurons_max  )
	data.encode_u32(   16, 1            )
	data.encode_float( 20, float(is_bias_used))
	data.encode_u32(   24, RUN          )
	device.buffer_update(common_data, 0, 28, data)

func run() -> void:
	var i : int = 1
	while i < layers_size:
		set_layer(i)
		submit(RUN)
		i += 1

func train(laps : int = 1) -> void:
	var iteration = 0
	while iteration < laps:
		var i : int = last_layer
		while i > 0:
			set_layer(i)
			submit(FIND_DELTAS)
			i -= 1
		
		i = 0
		while i < last_layer:
			set_layer(i)
			submit(TRAIN)
			i += 1
		if is_bias_used:
			i = 0
			while i < last_layer:
				set_layer(i)
				submit(BIAS_TRAIN)
				i += 1
		iteration += 1
		if iteration != laps:
			run()

func get_output() -> Array:
	return device.buffer_get_data(neurons_storage, last_layer * neurons_max * 4, layers[last_layer] * 4).to_float32_array()

func print_output() -> void:
	print(get_output())

func set_input(input : Array) -> void:
	assert(input.size() == layers[0], "GPUNNET set_input LINE " + str(get_stack()[0]["line"]) + ": Size of provided \"input\" doesn't match with the size of neural network's input. Provided input size: " + str(input.size()) + ". Neural network's input size: " + str(layers[0]) + ".")
	device.buffer_update(neurons_storage, 0, layers[0] * 4, PackedFloat32Array(input).to_byte_array())

func set_desired_output(output : Array) -> void:
	assert(output.size() == layers[last_layer], "GPUNNET set_desired_output LINE " + str(get_stack()[0]["line"]) + ": Size of provided \"output\" doesn't match with the size of neural network's output. Provided output size: " + str(output.size()) + ". Neural network's output size: " + str(layers[last_layer]) + ".")
	device.buffer_update(desired_output_storage, 0, layers[last_layer] * 4, PackedFloat32Array(output).to_byte_array())

func free_objects() -> void:
	device.free_rid(common_data)
	device.free_rid(bias_storage)
	device.free_rid(deltas_storage)
	device.free_rid(neurons_storage)
	device.free_rid(weights_storage)
	device.free_rid(structure_storage)
	device.free_rid(desired_output_storage)
	
	var i : int = 0; while i < 4:
		device.free_rid(pipeline[i])
		RenderingServer.free_rid(uniform_set[i])
		device.free_rid(shader[i]); i += 1
	
	device.free()

func free() -> void:
	free_objects()

func get_neurons(layer : int) -> Array:
	return device.buffer_get_data(neurons_storage, layer * neurons_max * 4, layers[layer] * 4).to_float32_array()

func print_neurons(layer : int) -> void:
	print(get_neurons(layer))

func get_weights(layer : int) -> Array:
	var weights : PackedFloat32Array = PackedFloat32Array([])
	var i : int = 0
	while i < layers[layer]:
		weights += device.buffer_get_data(weights_storage, (neurons_max * neurons_max * layer + i * neurons_max) * 4, layers[layer + 1] * 4).to_float32_array()
		i += 1
	return weights

func print_weights(layer : int) -> void:
	var data = get_weights(layer)
	var i : int = 0
	while i < layers[layer]:
		var weights : Array = data.slice(i * layers[layer + 1], i * layers[layer + 1] + layers[layer + 1])
		print("layer: ", layer, "; neuron: ", i, "; weights: ", weights)
		i += 1

func save_data(file_name : String) -> int:
	var file := FileAccess.open(full_path(file_name), FileAccess.WRITE)
	if ! file.is_open(): return ERR_FILE_CANT_OPEN
	file.store_8(1)
	file.store_8(is_bias_used)
	file.store_8(1)
	file.store_64(layers_size)
	for layer in layers:
		file.store_64(layer)
	var weights : Array = []
	var i : int = 0; while i < last_layer:
		weights.append_array(get_weights(i)); i += 1
	for weight in weights:
		file.store_double(weight)
	if is_bias_used:
		var biases : Array = []
		i = 1; while i < layers_size:
			biases.append_array(get_biases(i))
			i += 1
		for bias in biases:
			file.store_double(bias)
	file.close()
	return OK

func load_data(file_name : String) -> int:
	if is_corrupted(file_name): return ERR_FILE_CORRUPT
	var file := FileAccess.open(full_path(file_name), FileAccess.READ)
	if ! file.is_open(): return ERR_CANT_OPEN
	file.get_8()
	var bias : bool = file.get_8()
	file.get_8()
	var size : int = file.get_64()
	var structure : Array[int] = []
	structure.resize(size)
	var i : int = 0
	while i < size:
		structure[i] = file.get_64(); i += 1
	
	var weights : PackedFloat32Array = PackedFloat32Array([])
	var nmax : int = structure.max()
	weights.resize(nmax * (size - 1) * nmax )
	
	var d3 : Callable = func(x,y,z) -> int:
		return x * nmax * nmax + y * nmax + z
	
	var good : Callable = func(x,y,z) -> bool:
		if y < structure[x]:
			if z < structure[x + 1]:
				return true
		return false
	
	i = 0
	var j : int = 0
	var k : int = 0
	while i < size - 1:
		j = 0
		while j < nmax:
			k = 0
			while k < nmax:
				if good.call(i,j,k):
					weights[d3.call(i,j,k)] = file.get_double()
				k += 1
			j += 1
		i += 1
	
	var biases : PackedFloat32Array = PackedFloat32Array([])
	if bias:
		biases.resize((size - 1) * nmax )
		i = 0
		while i < size - 1:
			j = 0
			while j < nmax:
				biases[i * nmax + j] = file.get_double()
				j += 1
			i += 1
	
	var buffer : GPUNNET = GPUNNET.new(structure, 1.0, bias)
	buffer.device.buffer_update(buffer.weights_storage, 0, weights.size() * 4, weights.to_byte_array())
	buffer.device.buffer_update(buffer.bias_storage, 0, biases.size() * 4, biases.to_byte_array())
	assign(buffer)
	
	return OK

func is_same_structure(buffer : GPUNNET) -> bool:
	if layers != buffer.layers: return false
	if is_bias_used != buffer.is_bias_used: return false
	return true

func is_same_structure_file(file_name : String) -> bool:
	if is_corrupted(file_name): return false
	var file := FileAccess.open(full_path(file_name), FileAccess.READ)
	if ! file.is_open(): return false
	file.get_8()
	if file.get_8() != int(is_bias_used): return false
	file.get_8()
	if file.get_64() != layers_size: return false
	var i : int = 0; while i < layers_size:
		if layers[i] != file.get_64(): return false
		i += 1
	file.close()
	return true

static func is_corrupted(file_name : String) -> bool:
	if ! FileAccess.file_exists(full_path(file_name)): true
	var file := FileAccess.open(full_path(file_name), FileAccess.READ)
	if ! file.is_open(): return true
	var size : int = file.get_length()
	if size < 35: return true
	file.get_8()
	var bias : bool = file.get_8()
	file.get_8()
	var the_layers_size : int = file.get_64(); if the_layers_size < 2: return true
	var the_layers : Array[int] = []
	the_layers.resize(the_layers_size)
	var i : int = 0; while i < the_layers_size:
		the_layers[i] = file.get_64();
		if the_layers[i] < 1: return true
		i += 1
	if file.eof_reached(): return true
	var biases : int = 0
	if bias:
		i = 1; while i < the_layers_size:
			biases += the_layers[i]; i += 1
	var weights : int = 0
	i = 0; while i < the_layers_size - 1:
		weights += the_layers[i] * the_layers[i + 1]; i += 1
	var total : int = biases * 8 + weights * 8 + the_layers_size * 8 + 11
	if file.get_length() == total: return false
	print("file size: ", file.get_length(), " required size: ", total)
	return true

static func full_path(file_name : String) -> String:
	if file_name.begins_with("res://") or file_name.begins_with("user://"):
		return file_name
	return "res://addons/neural_network/data/" + file_name

func get_biases(layer : int) -> Array:
	assert(layer > 0, "There is no biases for layer " + str(layer))
	device.buffer_get_data(bias_storage, (layer - 1) * neurons_max * 4, layers[layer] * 4).to_float32_array()
	return []

func print_biases(layer : int) -> void:
	print(get_biases(layer))

func assign(buffer : GPUNNET) -> void:
	free_objects()
	_init(buffer.layers, buffer.learning_rate, buffer.is_bias_used)
	device.buffer_update(neurons_storage, 0, layers[0] * 4, buffer.get_neurons(0))
	device.buffer_update(desired_output_storage, 0, layers[last_layer] * 4, buffer.device.buffer_get_data(buffer.desired_output_storage))
	device.buffer_update(weights_storage, 0, last_layer * neurons_max * neurons_max * 4, buffer.device.buffer_get_data(buffer.weights_storage))
	if is_bias_used: device.buffer_update(bias_storage, 0, last_layer * neurons_max * 4, buffer.device.buffer_get_data(buffer.bias_storage))

func duplicate() -> GPUNNET:
	var buffer : GPUNNET = GPUNNET.new([1,1])
	buffer.assign(self)
	return buffer

func set_learning_rate(rate : float) -> void:
	device.buffer_update(common_data, 0, 4, PackedFloat32Array([rate]).to_byte_array())
