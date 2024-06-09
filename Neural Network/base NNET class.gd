@icon("res://addons/Neural Network/base NNET class.gd")
class_name BaseNNET

enum ActivationFunctions
{
	identity,
	binary_step,
	logistic,
	sigmoid = logistic,
	soft_step = logistic,
	tanh,
	ReLU,
	mish,
	swish,
	softmax # does's not work yet
}

enum Algorithm
{
	backpropogation,
	resilient_propagation
}

enum LossFunctions
{
	MSE,
	MAE,
	BCE,
	CCE, # don't forget to set act. function of last layer to softmax using Categorical Crossentropy
	Hinge_loss,
	Cosine_similarity_loss,
	LogCosh_loss
}

const version : String = "3.0.0"
const saving_path : String = "res://addons/Neural Network/data/"
const apzero : float = 0.000001

static func old_file_to_new(path: String) -> void:
	var file := FileAccess.open(full_path(path),FileAccess.READ)
	if not file.is_open():
		push_error("Couldn't open file: " + error_string(file.get_error())); return
	if file.get_buffer(4).get_string_from_utf8().begins_with("NNET"):
		print("File " + path + " was already updated")
	file.seek(0); file.get_8(); var use_bias = file.get_8()
	file.get_8(); var new_structure_size : int = file.get_64()
	var new_structure : Array = []; var i : int = 0
	while i < new_structure_size:
		new_structure.append(file.get_64())
		i += 1
	var file_nn : NNET = NNET.new(new_structure, 0.1, use_bias)
	i = 0; while i < file_nn.weights.size():
		var j : int = 0; while j < file_nn.weights[i].size():
			var k : int = 0; while k < file_nn.weights[i][j].size():
				file_nn.weights[i][j][k] = file.get_double()
				k += 1
			j += 1
		i += 1
	i = 0; while i < file_nn.biases.size():
		var j : int = 0; while j < file_nn.biases[i].size():
			file_nn.biases[i][j] = file.get_double()
			j += 1
		i += 1
	file.close()
	file_nn.save_data(path)
static func full_path(path : String) -> String:
	create_directory.call()
	if path.begins_with("res:")  \
	or path.begins_with("user:") :
		return path
	return saving_path + path
static var create_directory = func() -> void:
	if not DirAccess.dir_exists_absolute("res://addons"):
		DirAccess.make_dir_absolute("res://addons")
	if not DirAccess.dir_exists_absolute("res://addons/Neural Network"):
		DirAccess.make_dir_absolute("res://addons/Neural Network")
	if not DirAccess.dir_exists_absolute("res://addons/Neural Network/data"):
		DirAccess.make_dir_absolute("res://addons/Neural Network/data")
	create_directory = func() -> void: pass

func set_function(function : Variant, layer : int) : pass # sets activation function
func set_loss_function(function : Variant) : pass
func get_logits() : pass
func save_data(path : String, binary : bool = true) : pass
func load_data(path : String) : pass
func run() : pass
func predict(input : Array) -> Array: set_input(input); run(); return get_output()
func train() : pass
func is_same_structure(structure : Array) : pass
func is_same_structure_file(path : String) : pass
func set_input(input : Array) : pass
func set_target(desired_output : Array) : pass
func duplicate() : pass
func assign(neural_network) : pass
func get_output() : pass
func print_output() : pass
func print_info() : pass
func allocate_neurons() -> void: pass
func allocate_weights() -> void: pass
func allocate_deltas() -> void: pass
func allocate_biases() -> void: pass
func is_structure_valid(new_structure : Array) -> bool:
	var i : int = 0
	while i < new_structure.size():
		if not new_structure[i] is int:
			return false
		i += 1
	return new_structure.size() >= 2
func is_array_valid(array : Array, must_be_size : int) -> bool:
	var i : int = 0
	while i < array.size():
		if not array[i] is float:
			return false
		i += 1
	return array.size() == must_be_size

func use_backpropogation(learning_rate : float) -> void: pass
func use_resilient_propagation(update_value : float, mult_factor : float, reduc_fact : float, max_step : float, min_step : float) -> void: pass

