@icon("res://addons/Neural Network/base NNET class.gd")
class_name BNNET

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
	softmax,
	user_function
}

enum Algorithms
{
	resilient_propagation,
	adamW,
	nadam,
	NAG,
	yogi,
	adamax,
	adadelta,
	gradient_descent,
	no_algorithm
}

enum LossFunctions
{
	MSE,
	MAE,
	BCE,
	CCE,
	Hinge_loss,
	Cosine_similarity_loss,
	LogCosh_loss,
	Huber_loss,
	user_function
}

const version : String = "3.1.0"
const save_path : String = "res://addons/Neural Network/data/"
const apzero : float = 0.000001

static func old_file_to_new(path: String) -> void:
	var file = FileAccess.open(full_path(path), FileAccess.READ)
	# ------------ checking for errors
	if not file.is_open():
		push_error("Could not open file: ", error_string(file.get_error()))
	# ------------ preparing variables
	var string_file : String = file.get_buffer(11).get_string_from_utf8()
	var file_nn : NNET = NNET.new([1,1], false)
	# ------------ reseting position in the file
	file.seek(0)
	# ------------ updating file
	if string_file.begins_with("NNETB 3.0.0"):
		file.get_line()
		# ------------ getting necessary data
		var new_structure : Array = []
		new_structure.resize(file.get_16())
		var _use_bias : bool = bool(file.get_8())
		# ------------ getting layers
		var i : int = 0
		while i < new_structure.size():
			new_structure[i] = file.get_32()
			i += 1
		# ------------ initialising
		file_nn._init(new_structure, _use_bias)
		# ------------ loading weights
		i = 0
		while i < file_nn.weights.size():
			var j : int = 0
			while j < file_nn.weights[i].size():
				var k : int = 0
				while k < file_nn.weights[i][j].size():
					file_nn.weights[i][j][k] = file.get_double()
					k += 1
				j += 1
			i += 1
		if _use_bias:
			i = 0
			while i < file_nn.biases.size():
				var j : int = 0
				while j < file_nn.biases[i].size():
					file_nn.biases[i][j] = file.get_double()
					j += 1
				i += 1
		# ------------ changing file
		file.close()
		file_nn.save_binary(path)
		# ------------
	elif string_file.begins_with("NNETN 3.0.0"):
		file.get_line()
		# ------------ getting new structure
		var new_structure : Array = Array(file.get_line().get_slice(" : ", 1).split(" ")).map(func(string_neurons): return string_neurons.to_int())
		# ------------ skipping and initialising
		file.get_line()
		file_nn._init(new_structure, file.get_line().get_slice(" : ", 1).to_int() > 0)
		# ------------ loading weights
		var i : int = 0
		while i < file_nn.weights.size():
			var j : int = 0
			while j < file_nn.weights[i].size():
				var k : int = 0
				while k < file_nn.weights[i][j].size():
					file_nn.weights[i][j][k] = file.get_line().get_slice(" : ", 1).to_float()
					k += 1
				j += 1
			i += 1
		if file_nn.use_bias:
			i = 0
			while i < file_nn.biases.size():
				var j : int = 0
				while j < file_nn.biases[i].size():
					file_nn.biases[i][j] = file.get_line().get_slice(" : ", 1).to_float()
					j += 1
				i += 1
		# ------------ changing file
		file.close()
		file_nn.save_binary(path)
		# ------------
	else:
		# ------------ skipping and getting data
		file.get_8()
		var _use_bias : bool = bool(file.get_8())
		file.get_8()
		var new_structure : Array = []
		new_structure.resize(file.get_64())
		# ------------ getting layers
		var i : int = 0
		while i < new_structure.size():
			new_structure[i] = file.get_64()
			i += 1
		# ------------ initialising
		file_nn._init(new_structure, _use_bias)
		# ------------ loading weights
		i = 0
		while i < file_nn.weights.size():
			var j : int = 0
			while j < file_nn.weights[i].size():
				var k : int = 0
				while k < file_nn.weights[i][j].size():
					file_nn.weights[i][j][k] = file.get_double()
					k += 1
				j += 1
			i += 1
		if _use_bias:
			i = 0
			while i < file_nn.biases.size():
				var j : int = 0
				while j < file_nn.biases[i].size():
					file_nn.biases[i][j] = file.get_double()
					j += 1
				i += 1
		# ------------ changing file
		file.close()
		file_nn.save_binary(path)
		# ------------
	# ------------ closing the file
	file.close()
	# ------------
static func full_path(path : String) -> String:
	create_directory.call()
	if path.begins_with("res:")  \
	or path.begins_with("user:") :
		return path
	return save_path + path
static var create_directory = func() -> void:
	if not DirAccess.dir_exists_absolute("res://addons"):
		DirAccess.make_dir_absolute("res://addons")
	if not DirAccess.dir_exists_absolute("res://addons/Neural Network"):
		DirAccess.make_dir_absolute("res://addons/Neural Network")
	if not DirAccess.dir_exists_absolute("res://addons/Neural Network/data"):
		DirAccess.make_dir_absolute("res://addons/Neural Network/data")
	create_directory = func() -> void: pass

func is_structure_valid(new_structure : Array) -> bool:
	var i : int = 0
	while i < new_structure.size():
		if not new_structure[i] is int and new_structure[i] > 0:
			return false
		i += 1
	return new_structure.size() >= 2

