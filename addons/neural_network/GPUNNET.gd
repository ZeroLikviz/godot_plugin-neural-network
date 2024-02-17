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

var pipeline     : RID
var uniform_set  : RID
var shader       : RID      

var learning_rate : float
var layers : Array[int]
var last_layer : int
var layers_size : int
var is_bias_used : bool
var neurons_max : int
var x_workgroups : int

func _init(layers_construction: Array[int] = [1,1], learning_rate_value: float = 1.0, use_bias: bool = true) -> void:
	for layer in layers_construction:
		assert(layer >= 1, "GPUNNET _init LINE 30: Amount of neurons in layer can't be less than 0")
	
	device = RenderingServer.create_local_rendering_device()
	shader = device.shader_create_from_spirv(load("res://addons/neural_network/GPUNNET.glsl").get_spirv())
	pipeline = device.compute_pipeline_create(shader)
	
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
	random_weights.resize(layers_size * neurons_max * neurons_max)
	i = 0; while i < random_weights.size(): random_weights[i] = randf_range(-1.0, 1.0); i += 1
	
	common_data            = device.storage_buffer_create(28)
	bias_storage           = device.storage_buffer_create(last_layer * neurons_max * 4)
	deltas_storage         = device.storage_buffer_create(last_layer * neurons_max * 4)
	weights_storage        = device.storage_buffer_create(layers_size * neurons_max * neurons_max * 4, random_weights.to_byte_array())
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
	
	uniform_set = device.uniform_set_create(uniforms, shader, 0)
	
	x_workgroups = int(floor(float(neurons_max) / 64.0)) + 1
	
	submit()

func create_uniform(rid : RID, binding : int) -> RDUniform:
	var uniform = RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding = binding
	uniform.add_id(rid)
	return uniform

func submit() -> void:
	var compute_list := device.compute_list_begin()
	device.compute_list_bind_compute_pipeline(compute_list, pipeline)
	device.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	device.compute_list_dispatch(compute_list, x_workgroups,1,1)
	device.compute_list_end()
	
	device.submit()
	device.sync()

func set_mode(mode : int) -> void:
	var data : PackedByteArray = PackedByteArray([]); data.resize(4)
	data.encode_u32(0, mode)
	device.buffer_update(common_data, 24, 4, data)

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
	set_mode(RUN)
	var i : int = 1
	while i < layers_size:
		set_layer(i)
		submit()
		i += 1

func train(laps : int = 1) -> void:
	var iteration = 0
	while iteration < laps:
		set_mode(FIND_DELTAS)
		var i : int = last_layer
		while i > 0:
			set_layer(i)
			submit()
			i -= 1
		
		set_mode(TRAIN)
		i = 0
		while i < last_layer:
			set_layer(i)
			submit()
			i += 1
		if is_bias_used:
			set_mode(BIAS_TRAIN)
			i = 0
			while i < last_layer:
				set_layer(i)
				submit()
				i += 1
		iteration += 1
		if iteration != laps: run()

func get_output() -> Array:
	return device.buffer_get_data(neurons_storage, last_layer * neurons_max * 4, layers[last_layer] * 4).to_float32_array()

func print_output() -> void:
	print(get_output())

func set_input(input : Array[float]) -> void:
	assert(input.size() == layers[last_layer], "GPUNNET set_input LINE 157: Size of provided \"input\" doesn't match with the size of neural network's input. Provided input size: " + str(input.size()) + ". Neural network's input size: " + str(layers[0]) + ".")
	device.buffer_update(neurons_storage, 0, layers[0] * 4, PackedFloat32Array(input).to_byte_array())

func set_desired_output(output : Array[float]) -> void:
	assert(output.size() == layers[last_layer], "GPUNNET set_desired_output LINE 161: Size of provided \"output\" doesn't match with the size of neural network's output. Provided output size: " + str(output.size()) + ". Neural network's output size: " + str(layers[last_layer]) + ".")
	device.buffer_update(desired_output_storage, 0, layers[last_layer] * 4, PackedFloat32Array(output).to_byte_array())

func free_objects() -> void:
	device.free_rid(common_data)
	device.free_rid(bias_storage)
	device.free_rid(deltas_storage)
	device.free_rid(neurons_storage)
	device.free_rid(weights_storage)
	device.free_rid(structure_storage)
	device.free_rid(desired_output_storage)
	
	device.free_rid(pipeline)
	device.free_rid(shader)
	
	device.free()

func free() -> void:
	free_objects()

func get_neurons() -> Array:
	var neurons : PackedFloat32Array = device.buffer_get_data(neurons_storage, 0, layers[0] * 4).to_float32_array()
	var i : int = 1
	while i < layers_size:
		neurons += device.buffer_get_data(neurons_storage, i * neurons_max * 4, layers[i] * 4).to_float32_array()
		i += 1
	return neurons

func print_neurons() -> void:
	print(get_neurons())
