@icon("res://addons/neural_network/GPUNNET icon.png")
class_name GPUNNET

enum {run = 0, find_deltas = 1, train = 2}

var device                 : RenderingDevice

var common_data            : RID
var bias_storage           : RID
var deltas_storage         : RID
var weights_storage        : RID
var structure_storage      : RID
var output_neurons_storage : RID
var desired_output_storage : RID

var uniform_set  : RID

var learning_rate : float
var layers : Array[int]
var last_layer : int
var layers_size : int
var is_bias_used : bool

func _init(layers_construction: Array[int] = [1,1], learning_rate_value: float = 1.0, use_bias: bool = true) -> void:
	learning_rate = learning_rate_value
	last_layer    = layers_construction.size() - 1
	layers_size   = layers_construction.size()
	layers        = layers_construction
	is_bias_used = use_bias
	assert(layers_size >= 2, "structure of neural network should have at least 2 layers") 
	var i : int = 0
	while i < layers_size:
		assert(layers[i] >= 1, "layer size should be equal or more than 1") 
		i += 1
	
	device = RenderingServer.create_local_rendering_device()
	var shader = device.shader_create_from_spirv(load("res://addons/neural_network/GPUNNET.glsl").get_spirv())
	var pipeline = device.compute_pipeline_create(shader)
	
	var x_component = layers_size
	var y_component = layers.max()
	
	var random_weights : PackedByteArray = PackedByteArray([])
	random_weights.resize(x_component * (y_component ** 2) * 4)
	i = 0; while i < x_component * (y_component ** 2): random_weights.encode_float(i * 4, randf_range(-1.0, 1.0)); i += 1
	
	common_data            = device.storage_buffer_create(5                                * 4)
	bias_storage           = device.storage_buffer_create((x_component - 1) * y_component  * int(use_bias) * 4)
	deltas_storage         = device.storage_buffer_create((x_component - 1) * y_component        * 4)
	weights_storage        = device.storage_buffer_create(x_component * (y_component ** 2) * 4, random_weights)
	structure_storage      = device.storage_buffer_create(layers_size                      * 4)
	output_neurons_storage = device.storage_buffer_create(x_component * y_component        * 4)
	desired_output_storage = device.storage_buffer_create(y_component                      * 4)
	
	var uniforms : Array[RDUniform] = []; i = 0
	uniforms.append(initialize_uniform(common_data           , 0))
	uniforms.append(initialize_uniform(bias_storage          , 1))
	uniforms.append(initialize_uniform(deltas_storage        , 2))
	uniforms.append(initialize_uniform(weights_storage       , 3))
	uniforms.append(initialize_uniform(structure_storage     , 4))
	uniforms.append(initialize_uniform(output_neurons_storage, 5))
	uniforms.append(initialize_uniform(desired_output_storage, 6))
	
	var uniform_set = device.uniform_set_create(uniforms, shader, 0)
	
	fill_data()
	
	var compute_list = device.compute_list_begin()
	device.compute_list_bind_compute_pipeline( compute_list, pipeline                   )
	device.compute_list_bind_uniform_set(      compute_list, uniform_set, 0             )
	device.compute_list_dispatch(              compute_list, x_component - 1, y_component, 1)
	device.compute_list_end()

func initialize_uniform(rid : RID, binding : int) -> RDUniform:
	var uniform          = RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding      = binding
	uniform.add_id(rid)
	return uniform

func fill_data() -> void:
	var common_data_pba : PackedByteArray = PackedByteArray([])
	common_data_pba.resize(20)
	common_data_pba.encode_float(  0, learning_rate        )
	common_data_pba.encode_u32  (  4, last_layer           )
	common_data_pba.encode_u32  (  8, layers_size          )
	common_data_pba.encode_float( 12, float(is_bias_used)  )
	common_data_pba.encode_u32  ( 16, run                  )
	device.buffer_update(common_data, 0, 20, common_data_pba)
	
	var layers_pba : PackedByteArray = PackedByteArray([]); layers_pba.resize(layers_size * 4)
	var i = 0; while i < layers_size: layers_pba.encode_s32(i * 4, layers[i]); i += 1
	device.buffer_update(structure_storage, 0, layers_size * 4, layers_pba)

func set_mode(mode : int):
	var pba : PackedByteArray = PackedByteArray([])
	pba.resize(4); pba.encode_s32(0, mode)
	device.buffer_update(structure_storage, 12, 4, pba)
