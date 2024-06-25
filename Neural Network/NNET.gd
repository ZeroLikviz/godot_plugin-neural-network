@icon("res://addons/Neural Network/base NNET class.gd")
extends BNNET
class_name NNET

#region Variables

var aa : Array = [] # Algorithm additions (momentums for example)
var structure : Array = []
var weights : Array = []
var neurons_in : Array = []
var neurons_out : Array = []
var biases : Array = []
var neuron_deltas : Array = []
var weight_deltas : Array = []
var bias_deltas : Array = []
var wapply_deltas : Array = [] # buffer deltas. Deltas for updating weights
var bapply_deltas : Array = [] # buffer deltas. Deltas for updating biases
var target : Array = []
var uf : Array = []
var f : Array = []
var df : Array = []
var lr : float = 1.0 # learning rate
var use_bias : bool
var lf : Callable # loss function
var dlayer : Array = []
var algorithm : BNNET.Algorithms = BNNET.Algorithms.no_algorithm
var batch_size : int = 1
var fd : Array = [] #functions data

#endregion

#region Enums

enum Adam
{
	beta1,
	beta2,
	weight_decay,
	m1,
	m2,
	bm1,
	bm2
}

enum Rprop
{
	ep, # eta plus
	em, # eta minus
	max_step,
	min_step,
	wg, # weight gradients
	bg, # bias gradients
	uv # update value
}

enum NAG
{
	momentum, # velocity
	bmomentum, # bias velocity
	beta
}

enum Adadelta
{
	wgs, # weight gradients squared
	bgs, # bias gradients squared
	wds, # weight deltas squared
	bds, # bias deltas squared
	wd, # weight deltas
	bd, # bias deltas
	df # damping factor
}

#endregion

#region Arrays

func allocate3(array : Array) -> void:
	array.resize(structure.size() - 1)
	var i : int = 0
	while i < structure.size() - 1:
		array[i] = []
		array[i].resize(structure[i])
		var j : int = 0
		while j < structure[i]:
			array[i][j] = []
			array[i][j].resize(structure[i + 1])
			j += 1
		i += 1

func allocate2(array : Array, offset : int = 0) -> void:
	array.resize(structure.size() + offset)
	var i : int = 0
	while i < structure.size() + offset:
		array[i] = []
		array[i].resize(structure[i - offset])
		i += 1

func fill_rand3(array : Array, min : float = -1.0, max : float = 1.0) -> void:
	var i : int = 0
	while i < array.size():
		var j : int = 0
		while j < array[i].size():
			var k : int = 0
			while k < array[i][j].size():
				array[i][j][k] = randf_range(min, max)
				k += 1
			j += 1
		i += 1

func fill_rand2(array : Array, min : float = -1.0, max : float= 1.0) -> void:
	var i : int = 0
	while i < array.size():
		var j : int = 0
		while j < array[i].size():
			array[i][j] = randf_range(min, max)
			j += 1
		i += 1

func fill_zero2(array : Array) -> void:
	fill2(array, 0.0)
 
func fill_zero3(array : Array) -> void:
	fill3(array, 0.0)

func fill2(array : Array, value) -> void:
	var i : int = 0
	while i < array.size():
		var j : int = 0
		while j < array[i].size():
			array[i][j] = value
			j += 1
		i += 1
 
func fill3(array : Array, value) -> void:
	var i : int = 0
	while i < array.size():
		var j : int = 0
		while j < array[i].size():
			var k : int = 0
			while k < array[i][j].size():
				array[i][j][k] = value
				k += 1
			j += 1
		i += 1

func add_arrays3(c1 : float, c2 : float, array1 : Array, array2 : Array) -> void: # adds deltas to bdeltas
	var layer : int = 0
	while layer < array1.size():
		var neuron : int = 0
		while neuron < array1[layer].size():
			var forward_neuron : int = 0
			while forward_neuron < array1[layer][neuron].size():
				array1[layer][neuron][forward_neuron] = array1[layer][neuron][forward_neuron] * c1 + array2[layer][neuron][forward_neuron] * c2
				forward_neuron += 1
			neuron += 1
		layer += 1

func add_arrays2(c1 : float, c2 : float, array1 : Array, array2 : Array) -> void: # adds deltas to bdeltas
	var layer : int = 0
	while layer < array1.size():
		var neuron : int = 0
		while neuron < array1[layer].size():
			array1[layer][neuron] = array1[layer][neuron] * c1 + array2[layer][neuron] * c2
			neuron += 1
		layer += 1

static func deep_copy(array : Array, deep_steps : int = 16) -> Variant:
	if deep_steps <= 0:
		return null
	var copy : Array = []
	copy.resize(array.size())
	var i : int = 0
	while i < array.size():
		if not array[i] is Array:
			copy[i] = array[i]
		else:
			copy[i] = deep_copy(array[i], deep_steps - 1)
		i += 1
	return copy

#endregion

#region Functions

func set_function(function : Variant, from : int, to : int = from) -> void:
	if from < 0:
		push_error("There is no negative layers, don't use negative numbers")
		return
	if to >= structure.size():
		push_error("You exceeded number of layers by ", to - structure.size() + 1 ,".Remember: From and To are both inclusive")
		return
	if function is Callable or function is BNNET.ActivationFunctions:
		var layer : int = from
		while layer <= to:
			specify_function(function, layer)
			layer += 1
	else:
		push_error("Function must be Callable or belong to BNNET.ActivationFunctions")

func specify_function(function : Variant, layer : int) -> void:
	if function is Callable:
		fd[layer] = BNNET.ActivationFunctions.user_function
		uf[layer] = function
		f[layer] = func() -> void:
			var i : int = 0
			while i < neurons_in[layer].size():
				neurons_in[layer][i] = uf[layer].call(neurons_in[layer][i])
				i += 1
		df[layer] = func() -> void:
			var i : int = 0
			while i < neurons_in[layer].size():
				dlayer[i] = uf[layer].call(neurons_in[layer][i] + apzero) - uf[layer].call(neurons_in[layer][i]) / apzero
				i += 1
	elif function is BNNET.ActivationFunctions:
		if function != BNNET.ActivationFunctions.user_function:
			fd[layer] = function
		match function:
			BNNET.ActivationFunctions.identity:
				f[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = neurons_in[layer][i]
						i += 1
				df[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						dlayer[i] = 1.0
						i += 1
			BNNET.ActivationFunctions.binary_step:
				f[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = float(neurons_in[layer][i] >= 0)
						i += 1
				df[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						dlayer[i] = 0.0
						i += 1
			BNNET.ActivationFunctions.logistic:
				f[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = 1.0 / (1.0 + exp(-neurons_in[layer][i]))
						i += 1
				df[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						dlayer[i] = neurons_out[layer][i] * ( 1.0 - neurons_out[layer][i] )
						i += 1
			BNNET.ActivationFunctions.tanh:
				f[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = 2.0 / (1.0 + exp(-neurons_in[layer][i])) - 1.0
						i += 1
				df[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						var expx = 2.0 / (1.0 + exp(-neurons_in[layer][i])) - 1.0
						dlayer[i] = 1.0 - expx * expx
						i += 1
			BNNET.ActivationFunctions.ReLU:
				f[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = neurons_in[layer][i] * int(neurons_in[layer][i] >= 0)
						i += 1
				df[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						dlayer[i] = float(neurons_in[layer][i] >= 0)
						i += 1
			BNNET.ActivationFunctions.mish:
				f[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = neurons_in[layer][i] * \
						tanh(log(1.0 + exp(neurons_in[layer][i])))
						i += 1
				df[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						var expx : float = exp(neurons_in[layer][i])
						var x4 : float = neurons_in[layer][i] * 4
						var divider : float = expx * (expx + 2) + 2
						divider *= divider
						dlayer[i] = expx * ( 4 + x4 + expx * (expx * (expx + 4) + x4 + 6) ) / divider
						i += 1
			BNNET.ActivationFunctions.swish:
				f[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = neurons_in[layer][i] / (1.0 + exp(-neurons_in[layer][i]))
						i += 1
				df[layer] = func() -> void:
					var i : int = 0
					while i < neurons_in[layer].size():
						var sx : float = 1.0 / (1.0 + exp(-neurons_in[layer][i]))
						dlayer[i] = sx + neurons_in[layer][i] * sx * (1.0 - sx)
						i += 1
			BNNET.ActivationFunctions.softmax:
				f[layer] = func() -> void:
					var sum : float = 0.0
					var i : int = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] = exp(neurons_in[layer][i])
						sum += neurons_out[layer][i]
						i += 1
					i = 0
					while i < neurons_in[layer].size():
						neurons_out[layer][i] /= sum
						i += 1
				df[layer] = func() -> void:
					var sum = exp(neurons_in[layer][0]) / neurons_out[layer][0]
					var i : int = 0
					while i < neurons_in[layer].size():
						dlayer[i] = ( neurons_out[layer][i] - \
						exp(neurons_in[layer][i] + apzero) / sum ) / apzero
						i += 1
	else:
		push_error("Function must be Callable or belong to BNNET.ActivationFunctions")

func set_loss_function(function : Variant) -> void:
	if function is Callable:
		fd[f.size()] = BNNET.LossFunctions.user_function
		lf = function
		# ------------- test
		var array : Array = []
		array.resize(f.size())
		array.fill(1.2345)
		var test_error : float = lf.call(array, array)
		var i : int = 0
		while i < f.size():
			if not is_equal_approx(array[i], 1.2345):
				push_error("It's not allowed for loss function to change output/target values. Change your function (If your function doesn't change anything, then remove this chunk of code")
				i += 1
		# -------------
	elif function is BNNET.LossFunctions:
		if function != BNNET.LossFunctions.user_function:
			fd[f.size()] = function
		match function:
			BNNET.LossFunctions.MSE:
				lf = func(outputs, targets) -> float:
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						sum += (outputs[i] - targets[i]) * (outputs[i] - targets[i])
						i += 1
					sum /= outputs.size()
					return sum
			BNNET.LossFunctions.MAE:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						var part_sum : float = outputs[i] - targets[i]
						sum += int(part_sum > 0) * part_sum + int(part_sum < 0) * -part_sum
						i += 1
					sum /= outputs.size()
					return sum
			BNNET.LossFunctions.BCE:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						sum += targets[i] * log(outputs[i]) + (1.0 - targets[i]) * log(1.0 - outputs[i])
						i += 1
					sum /= outputs.size()
					return -sum
			BNNET.LossFunctions.CCE:
				lf = func(outputs, targets):
					var i : int = 0
					var sum : float = 0.0
					while i < outputs.size():
						sum += targets[i] * log(outputs[i])
						i += 1
					sum /= outputs.size()
					return -sum
			BNNET.LossFunctions.Hinge_loss:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						var sum_part : float = 1.0 - targets[i] * outputs[i]
						sum += int(sum_part > 0) * sum_part
						i += 1
					sum /= outputs.size()
					return sum
			BNNET.LossFunctions.Cosine_similarity_loss:
				lf = func(outputs, targets):
					var length_o : float = 0
					var length_t : float = 0
					var i : int = 0
					while i < outputs.size():
						length_o += outputs[i] * outputs[i]
						length_t += targets[i] * targets[i]
						i += 1
					length_o = sqrt(length_o)
					length_t = sqrt(length_t)
					var similarity : float
					i = 0
					while i < outputs.size():
						similarity += (outputs[i] / length_o) * (targets[i] / length_t)
						i += 1
					return similarity
			BNNET.LossFunctions.LogCosh_loss:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						sum += log(cosh(outputs[i] - targets[i]))
						i += 1
					sum /= outputs.size()
					return sum
	else:
		push_error("Function must be Callable or belong to BNNET.LossFunctions")

#endregion

#region Delta

func compute_deltas_last_layer() -> void:
	df[f.size() - 1].call()
	var error = lf.call(neurons_out[f.size() - 1], target)
	var last_layer_neurons_copy : Array = neurons_in[f.size() - 1].duplicate()
	var i : int = 0
	while i < structure[f.size() - 1]:
		neurons_in[f.size() - 1][i] += apzero
		f[f.size() - 1].call()
		neuron_deltas[f.size() - 1][i] = dlayer[i] * (lf.call(neurons_out[f.size() - 1], target) - error) / apzero
		neurons_in[f.size() - 1][i] = last_layer_neurons_copy[i]
		i += 1
	f[f.size() - 1].call()

func compute_deltas_hidden_layers() -> void:
	var layer : int = neuron_deltas.size() - 2
	while layer > 0:
		df[layer].call()
		var neuron : int = 0
		while neuron < neuron_deltas[layer].size():
			neuron_deltas[layer][neuron] = 0.0
			var forward_neuron : int = 0
			while forward_neuron < neuron_deltas[layer + 1].size():
				neuron_deltas[layer][neuron] += neuron_deltas[layer + 1][forward_neuron] * weights[layer][neuron][forward_neuron]
				forward_neuron += 1
			neuron_deltas[layer][neuron] *= dlayer[neuron]
			neuron += 1
		layer -= 1

func compute_weight_deltas() -> void:
	var layer : int = 0
	while layer < weights.size():
		var neuron : int = 0
		while neuron < weights[layer].size():
			var forward_neuron : int = 0
			while forward_neuron < weights[layer][neuron].size():
				weight_deltas[layer][neuron][forward_neuron] = neuron_deltas[layer + 1][forward_neuron] * neurons_out[layer][neuron]
				forward_neuron += 1
			neuron += 1
		layer += 1
	if use_bias:
		layer = 0
		while layer < biases.size():
			var neuron : int = 0
			while neuron < biases[layer].size():
				bias_deltas[layer][neuron] = neuron_deltas[layer + 1][neuron]
				neuron += 1
			layer += 1

#endregion

#region Propagation

func propagate_forward() -> void:
	var layer : int = 1
	while layer < structure.size():
		var neuron : int = 0
		while neuron < structure[layer]:
			var backward_neuron : int = 0
			neurons_in[layer][neuron] = biases[layer - 1][neuron]
			while backward_neuron < structure[layer - 1]:
				neurons_in[layer][neuron] += neurons_out[layer - 1][backward_neuron] * weights[layer - 1][backward_neuron][neuron]
				backward_neuron += 1
			neuron += 1
		f[layer].call()
		layer += 1

func propagate_backward() -> void:
	compute_deltas_last_layer()
	compute_deltas_hidden_layers()
	compute_weight_deltas()

#endregion

#region Init

func _init(architecture : Array, use_biases : bool) -> void:
	if not is_structure_valid(architecture):
		push_error("Architecture must contain only positive integers and have at least 2 layers!")
		return
	structure = architecture
	use_bias = use_biases
	target.resize(structure[structure.size() - 1])
	target.fill(0.0)
	dlayer.resize(structure.max())
	dlayer.fill(0.0)
	allocate3(weights)
	fill_rand3(weights)
	allocate2(neuron_deltas)
	fill_zero2(neuron_deltas)
	allocate3(wapply_deltas)
	fill_zero3(wapply_deltas)
	allocate3(weight_deltas)
	fill_zero3(weight_deltas)
	allocate2(biases, -1)
	fill_zero2(biases)
	if use_bias:
		fill_rand2(biases)
		allocate2(bias_deltas, -1)
		fill_zero2(bias_deltas)
		allocate2(bapply_deltas, -1)
		fill_zero2(bapply_deltas)
	allocate2(neurons_out)
	allocate2(neurons_in)
	fill_zero2(neurons_out)
	fill_zero2(neurons_in)
	f.resize(structure.size())
	df.resize(structure.size())
	uf.resize(structure.size())
	fd.resize(structure.size() + 1)
	set_function(BNNET.ActivationFunctions.logistic, 0, structure.size() - 1)
	set_loss_function(BNNET.LossFunctions.MSE)

func reinit() -> void:
	fill_rand3(weights)
	if use_bias:
		fill_rand2(biases)
	match algorithm:
		BNNET.Algorithms.adamW:
			use_Adam(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BNNET.Algorithms.adamax:
			use_Adamax(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BNNET.Algorithms.nadam:
			use_Nadam(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BNNET.Algorithms.yogi:
			use_Yogi(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BNNET.Algorithms.NAG:
			use_NAG(lr, aa[NAG.beta])
		BNNET.Algorithms.resilient_propagation:
			use_Rprop(aa[Rprop.uv], aa[Rprop.ep], aa[Rprop.em], aa[Rprop.max_step], aa[Rprop.min_step])
		BNNET.Algorithms.adadelta:
			use_Adadelta(aa[Adadelta.df])

func init_adam(beta_1 : float, beta_2 : float, weights_decay : float) -> void:
	if beta_1 > 1.0 or beta_1 < 0.0 or beta_2 > 1.0 or beta_2 < 0.0:
		push_error("Parameter beta must be in range from 0.0 to 1.0")
	if weights_decay < 0:
		push_error("Weight decay must not be negative number")
	aa.resize(7)
	aa[Adam.beta1] = beta_1
	aa[Adam.beta2] = beta_2
	aa[Adam.weight_decay] = weights_decay
	aa[Adam.m1] = []
	aa[Adam.m2] = []
	aa[Adam.bm1] = []
	aa[Adam.bm2] = []
	allocate3(aa[Adam.m1])
	allocate3(aa[Adam.m2])
	fill_zero3(aa[Adam.m1])
	fill_zero3(aa[Adam.m2])
	if use_bias:
		allocate2(aa[Adam.bm1], -1)
		allocate2(aa[Adam.bm2], -1)
		fill_zero2(aa[Adam.bm1])
		fill_zero2(aa[Adam.bm2])

func init_Rprop(update_value : float, eta_plus : float, eta_minus : float, mx_step : float, mn_step) -> void:
	if update_value < 0:
		push_error("Update value must be a positive number")
		return
	if eta_plus < 1.0:
		push_error("Eta plus must be greater than 1")
		return
	if eta_minus < 0.0 or eta_minus > 1.0:
		push_error("Eta minus must be in range from 0 to 1")
		return
	aa.resize(7)
	aa[Rprop.uv] = update_value
	aa[Rprop.ep] = eta_plus
	aa[Rprop.em] = eta_minus
	aa[Rprop.max_step] = mx_step
	aa[Rprop.min_step] = mn_step
	aa[Rprop.wg] = []
	aa[Rprop.bg] = []
	allocate3(aa[Rprop.wg])
	fill3(aa[Rprop.wg], update_value)
	if use_bias:
		allocate2(aa[Rprop.bg], -1)
		fill2(aa[Rprop.bg], update_value)

func init_NAG(beta : float) -> void:
	if beta < 0:
		push_error("Beta must be greater than zero")
	aa.resize(5)
	aa[NAG.beta] = beta
	aa[NAG.momentum] = []
	aa[NAG.bmomentum] = []
	allocate3(aa[NAG.momentum])
	fill_zero3(aa[NAG.momentum])
	if use_bias:
		allocate2(aa[NAG.bmomentum], -1)
		fill_zero2(aa[NAG.bmomentum])

func init_Adadelta(damping_factor : float) -> void:
	if damping_factor < 0.0 or damping_factor > 1.0:
		push_error("Damping factor must be in range from 0.0 to 1.0")
	aa.resize(7)
	aa[Adadelta.wgs] = []
	aa[Adadelta.bgs] = []
	aa[Adadelta.wds] = []
	aa[Adadelta.bds] = []
	aa[Adadelta.wd] = []
	aa[Adadelta.bd] = []
	aa[Adadelta.df] = damping_factor
	allocate3(aa[Adadelta.wd])
	allocate3(aa[Adadelta.wgs])
	allocate3(aa[Adadelta.wds])
	fill_zero3(aa[Adadelta.wd])
	fill_zero3(aa[Adadelta.wgs])
	fill_zero3(aa[Adadelta.wds])
	if use_bias:
		allocate2(aa[Adadelta.bd], -1)
		allocate2(aa[Adadelta.bgs], -1)
		allocate2(aa[Adadelta.bds], -1)
		fill_zero2(aa[Adadelta.bd])
		fill_zero2(aa[Adadelta.bgs])
		fill_zero2(aa[Adadelta.bds])

func kill_additions() -> void:
	aa.clear()

#endregion

#region Use

func use_gradient_descent(learning_rate : float) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	algorithm = BNNET.Algorithms.gradient_descent

func use_Adam(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_adam(beta_1, beta_2, weights_decay)
	algorithm = BNNET.Algorithms.adamW

func use_Nadam(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_adam(beta_1, beta_2, weights_decay)
	algorithm = BNNET.Algorithms.nadam

func use_Adamax(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_adam(beta_1, beta_2, weights_decay)
	algorithm = BNNET.Algorithms.nadam

func use_Yogi(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_adam(beta_1, beta_2, weights_decay)
	algorithm = BNNET.Algorithms.yogi

func use_Rprop(update_value : float = 0.1, eta_plus : float = 1.2, eta_minus : float = 0.5, maximum_step : float = 50.0, minimum_step = 0.000001) -> void:
	kill_additions()
	init_Rprop(update_value, eta_plus, eta_minus, maximum_step, minimum_step)
	algorithm = BNNET.Algorithms.resilient_propagation

func use_Adadelta(damping_factor : float = 0.9) -> void:
	kill_additions()
	init_Adadelta(damping_factor)
	algorithm = BNNET.Algorithms.adadelta

func use_NAG(learning_rate : float, beta : float) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_NAG(beta)
	algorithm = BNNET.Algorithms.NAG

#endregion

#region Update

func update_momentums_adam() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				aa[Adam.m1][i][j][k] = aa[Adam.beta1] * aa[Adam.m1][i][j][k] + ( 1.0 - aa[Adam.beta1] ) * weight_deltas[i][j][k]
				aa[Adam.m2][i][j][k] = aa[Adam.beta2] * aa[Adam.m2][i][j][k] + ( 1.0 - aa[Adam.beta2] ) * weight_deltas[i][j][k] * weight_deltas[i][j][k]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				aa[Adam.bm1][i][j] = aa[Adam.beta1] * aa[Adam.bm1][i][j] + ( 1.0 - aa[Adam.beta1] ) * bias_deltas[i][j]
				aa[Adam.bm2][i][j] = aa[Adam.beta2] * aa[Adam.bm2][i][j] + ( 1.0 - aa[Adam.beta2] ) * bias_deltas[i][j] * bias_deltas[i][j]
				j += 1
			i += 1

func update_momentums_yogi() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				aa[Adam.m1][i][j][k] = aa[Adam.beta1] * aa[Adam.m1][i][j][k] + ( 1.0 - aa[Adam.beta1] ) * weight_deltas[i][j][k]
				var gradient_sign : int = int(weight_deltas[i][j][k] * weight_deltas[i][j][k] - aa[Adam.m2][i][j][k] > 0)
				gradient_sign = gradient_sign + (1 - gradient_sign) * -1
				aa[Adam.m2][i][j][k] = aa[Adam.m2][i][j][k] + ( 1.0 - aa[Adam.beta2] ) * weight_deltas[i][j][k] * weight_deltas[i][j][k] * gradient_sign
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				aa[Adam.bm1][i][j] = aa[Adam.beta1] * aa[Adam.bm1][i][j] + ( 1.0 - aa[Adam.beta1] ) * bias_deltas[i][j]
				var gradient_sign : int = int(bias_deltas[i][j] * bias_deltas[i][j] - aa[Adam.bm2][i][j] > 0)
				gradient_sign = gradient_sign + (1 - gradient_sign) * -1
				aa[Adam.bm2][i][j] = aa[Adam.bm2][i][j] + ( 1.0 - aa[Adam.beta2] ) * bias_deltas[i][j] * bias_deltas[i][j] * gradient_sign
				j += 1
			i += 1

func update_momentums_adamax() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				aa[Adam.m1][i][j][k] = aa[Adam.beta1] * aa[Adam.m1][i][j][k] + ( 1.0 - aa[Adam.beta1] ) * weight_deltas[i][j][k]
				aa[Adam.m2][i][j][k] = aa[Adam.beta2] * aa[Adam.m2][i][j][k]
				var abs_grad : float = weight_deltas[i][j][k] * int(weight_deltas[i][j][k] >= 0) - weight_deltas[i][j][k] * int(weight_deltas[i][j][k] < 0)
				aa[Adam.m2][i][j][k] = int(aa[Adam.m2][i][j][k] > abs_grad) * aa[Adam.m2][i][j][k] + abs_grad * int(aa[Adam.m2][i][j][k] <= abs_grad)
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				aa[Adam.bm1][i][j] = aa[Adam.beta1] * aa[Adam.bm1][i][j] + ( 1.0 - aa[Adam.beta1] ) * bias_deltas[i][j]
				aa[Adam.bm2][i][j] = aa[Adam.beta2] * aa[Adam.bm2][i][j]
				var abs_grad : float = bias_deltas[i][j] * int(bias_deltas[i][j] >= 0) - bias_deltas[i][j] * int(bias_deltas[i][j] < 0)
				aa[Adam.bm2][i][j] = int(aa[Adam.bm2][i][j] > abs_grad) * aa[Adam.bm2][i][j] + abs_grad * int(aa[Adam.bm2][i][j] <= abs_grad)
				j += 1
			i += 1

func update_deltas_Rprop() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				var same_sign : int = int(weight_deltas[i][j][k] * aa[Rprop.wg][i][j][k] > 0)
				var sign : int = int(weight_deltas[i][j][k] > 0)
				sign = sign + (sign - 1)
				aa[Rprop.wg][i][j][k] = aa[Rprop.wg][i][j][k] * aa[Rprop.ep] * same_sign - (1 - same_sign) * aa[Rprop.em] * aa[Rprop.wg][i][j][k]
				var greater : int = int(aa[Rprop.wg][i][j][k] * sign > aa[Rprop.max_step])
				var lesser : int = int(aa[Rprop.wg][i][j][k] * sign < aa[Rprop.min_step]) 
				aa[Rprop.wg][i][j][k] = aa[Rprop.wg][i][j][k] * (1 - greater) + aa[Rprop.max_step] * greater * sign
				aa[Rprop.wg][i][j][k] = aa[Rprop.wg][i][j][k] * (1 - lesser) + aa[Rprop.min_step] * lesser * sign
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				var same_sign : int = int(bias_deltas[i][j] * aa[Rprop.bg][i][j] > 0)
				var sign : int = int(bias_deltas[i][j] > 0)
				sign = sign + (sign - 1)
				aa[Rprop.bg][i][j] = aa[Rprop.bg][i][j] * aa[Rprop.ep] * same_sign - (1 - same_sign) * aa[Rprop.em] * aa[Rprop.bg][i][j]
				var greater : int = int(aa[Rprop.bg][i][j] * sign > aa[Rprop.max_step])
				var lesser : int = int(aa[Rprop.bg][i][j] * sign < aa[Rprop.min_step]) 
				aa[Rprop.bg][i][j] = aa[Rprop.bg][i][j] * (1 - greater) + aa[Rprop.max_step] * greater * sign
				aa[Rprop.bg][i][j] = aa[Rprop.bg][i][j] * (1 - lesser) + aa[Rprop.min_step] * lesser * sign
				j += 1
			i += 1

func update_momentums_NAG() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				aa[NAG.momentum][i][j][k] = aa[NAG.beta] * aa[NAG.momentum][i][j][k] + lr * weight_deltas[i][j][k]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				aa[NAG.bmomentum][i][j] = aa[NAG.beta] * aa[NAG.bmomentum][i][j] + lr * bias_deltas[i][j]
				j += 1
			i += 1

func update_gradients_squared_Adadelta() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				aa[Adadelta.wgs][i][j][k] = aa[Adadelta.df] * aa[Adadelta.wgs][i][j][k] + (1.0 - aa[Adadelta.df]) * weight_deltas[i][j][k] * weight_deltas[i][j][k]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				aa[Adadelta.bgs][i][j] = aa[Adadelta.df] * aa[Adadelta.bgs][i][j] + (1.0 - aa[Adadelta.df]) * bias_deltas[i][j] * bias_deltas[i][j]
				j += 1
			i += 1

func update_deltas_squared_Adadelta() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				aa[Adadelta.wds][i][j][k] = aa[Adadelta.df] * aa[Adadelta.wds][i][j][k] + (1.0 - aa[Adadelta.df]) * aa[Adadelta.wd][i][j][k]  * aa[Adadelta.wd][i][j][k]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				aa[Adadelta.bds][i][j] = aa[Adadelta.df] * aa[Adadelta.bds][i][j] + (1.0 - aa[Adadelta.df]) * aa[Adadelta.bd][i][j]  * aa[Adadelta.bd][i][j]
				j += 1
			i += 1

func update_deltas_Adadelta() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				aa[Adadelta.wd][i][j][k] = -sqrt(aa[Adadelta.wds][i][j][k] + apzero) * weight_deltas[i][j][k] / sqrt(aa[Adadelta.wgs][i][j][k] + apzero)
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				aa[Adadelta.bd][i][j] = -sqrt(aa[Adadelta.bds][i][j] + apzero) * bias_deltas[i][j] / sqrt(aa[Adadelta.bgs][i][j] + apzero)
				j += 1
			i += 1

#endregion

#region Transfer

func momentums_to_apply_adam() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				wapply_deltas[i][j][k] = (aa[Adam.m1][i][j][k] / (1.0 - aa[Adam.beta1])) \
				/ (sqrt(aa[Adam.m2][i][j][k] / (1.0 - aa[Adam.beta2])) + BNNET.apzero) + weights[i][j][k] * aa[Adam.weight_decay]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				bapply_deltas[i][j] = (aa[Adam.bm1][i][j] / (1.0 - aa[Adam.beta1])) \
				/ (sqrt(aa[Adam.bm2][i][j] / (1.0 - aa[Adam.beta2])) + BNNET.apzero) + biases[i][j] * aa[Adam.weight_decay]
				j += 1
			i += 1

func momentums_to_apply_adamax() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				wapply_deltas[i][j][k] = (aa[Adam.m1][i][j][k] / (1.0 - aa[Adam.beta1])) \
				/ (aa[Adam.m2][i][j][k] / (1.0 - aa[Adam.beta2]) + BNNET.apzero) + weights[i][j][k] * aa[Adam.weight_decay]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				bapply_deltas[i][j] = (aa[Adam.bm1][i][j] / (1.0 - aa[Adam.beta1])) \
				/ (aa[Adam.bm2][i][j] / (1.0 - aa[Adam.beta2]) + BNNET.apzero) + biases[i][j] * aa[Adam.weight_decay]
				j += 1
			i += 1

func momentums_to_apply_nadam() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				wapply_deltas[i][j][k] = ( aa[Adam.beta1] * (aa[Adam.m1][i][j][k] / (1.0 - aa[Adam.beta1])) + weight_deltas[i][j][k] )\
				/ (sqrt(aa[Adam.m2][i][j][k] / (1.0 - aa[Adam.beta2])) + BNNET.apzero) + weights[i][j][k] * aa[Adam.weight_decay]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				bapply_deltas[i][j] = ( aa[Adam.beta1] * (aa[Adam.bm1][i][j] / (1.0 - aa[Adam.beta1])) + bias_deltas[i][j] ) \
				/ (sqrt(aa[Adam.bm2][i][j] / (1.0 - aa[Adam.beta2])) + BNNET.apzero) + biases[i][j] * aa[Adam.weight_decay]
				j += 1
			i += 1

func momentums_to_apply_NAG() -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				wapply_deltas[i][j][k] = aa[NAG.momentum][i][j][k]
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				bapply_deltas[i][j] = aa[NAG.bmomentum][i][j]
				j += 1
			i += 1

func zero_apply_grad() -> void:
	fill_zero3(wapply_deltas)
	if use_bias:
		fill_zero2(bapply_deltas)

func apply_to_general() -> void:
	add_arrays3(0.0, 1.0, weight_deltas, wapply_deltas)
	if use_bias:
		add_arrays2(0.0, 1.0, bias_deltas, bapply_deltas)

func add_to_apply(c1 : float, c2 : float, warray : Array, barray : Array) -> void:
	add_arrays3(c1, c2, wapply_deltas, warray)
	if use_bias:
		add_arrays2(c1, c2, bapply_deltas, barray)

#endregion

#region Algorithms

func apply_gradients(c : float) -> void:
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				weights[i][j][k] -= wapply_deltas[i][j][k] * c
				k += 1
			j += 1
		i += 1
	i = 0
	while i < biases.size() and use_bias:
		var j : int = 0
		while j < biases[i].size():
			biases[i][j] -= bapply_deltas[i][j] * c
			j += 1
		i += 1

func train(input_data : Array[Array], target_data : Array[Array]) -> void:
	if algorithm == BNNET.Algorithms.no_algorithm:
		push_error("Please set algorithm using use_gradient_descent(...), use_adam, use_resilient_propagation and etc.")
		return
	if input_data.size() == 0:
		push_error("Provide input data")
		return
	if input_data.size() < batch_size:
		push_error("Batch size is greater than number of elements in provided training data")
		return
	if input_data.size() != target_data.size():
		push_error("Number of elements in input data array doesn't match number of elements in target data array")
		return
	match algorithm:
		BNNET.Algorithms.gradient_descent:
			zero_apply_grad()
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			apply_gradients(lr)
		BNNET.Algorithms.adamW:
			zero_apply_grad()
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			apply_to_general()
			update_momentums_adam()
			momentums_to_apply_adam()
			apply_gradients(lr)
		BNNET.Algorithms.resilient_propagation:
			zero_apply_grad()
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			apply_to_general()
			update_deltas_Rprop()
			add_to_apply(0.0, 1.0, aa[Rprop.wg], aa[Rprop.bg])
			apply_gradients(1.0)
		BNNET.Algorithms.NAG:
			zero_apply_grad()
			# ------------ changing weights
			add_arrays3(1.0, -aa[NAG.beta], weights, aa[NAG.momentum])
			if use_bias:
				add_arrays2(1.0, -aa[NAG.beta], biases, aa[NAG.bmomentum])
			# ------------
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			# ------------ restoring weights
			add_arrays3(1.0, aa[NAG.beta], weights, aa[NAG.momentum])
			if use_bias:
				add_arrays2(1.0, aa[NAG.beta], biases, aa[NAG.bmomentum])
			# ------------
			apply_to_general()
			update_momentums_NAG()
			momentums_to_apply_NAG()
			apply_gradients(1.0)
		BNNET.Algorithms.nadam:
			zero_apply_grad()
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			apply_to_general()
			update_momentums_adam()
			momentums_to_apply_nadam()
			apply_gradients(lr)
		BNNET.Algorithms.adamax:
			zero_apply_grad()
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			apply_to_general()
			update_momentums_adamax()
			momentums_to_apply_adamax()
			apply_gradients(lr)
		BNNET.Algorithms.yogi:
			zero_apply_grad()
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			apply_to_general()
			update_momentums_yogi()
			momentums_to_apply_adam()
			apply_gradients(lr)
		BNNET.Algorithms.adadelta:
			zero_apply_grad()
			var i : int = 0
			while i < batch_size:
				var training_set : int = randi() % input_data.size() * int(batch_size != input_data.size()) \
				+ int(batch_size == input_data.size()) * i
				set_input(input_data[training_set])
				set_target(target_data[training_set])
				propagate_forward()
				propagate_backward()
				add_to_apply(1.0, 1.0 / batch_size, weight_deltas, bias_deltas)
				i += 1
			apply_to_general()
			update_gradients_squared_Adadelta()
			update_deltas_Adadelta()
			# ------------ updating weights
			add_to_apply(0.0, 1.0, aa[Adadelta.wd], aa[Adadelta.bd])
			apply_gradients(-1.0 * batch_size)
			# ------------
			update_deltas_squared_Adadelta()

#endregion

#region Checker

func check_learning_rate(learning_rate : float) -> void:
	if learning_rate < 0:
		push_error("Learning rate must be above zero")
		return
	lr = learning_rate

#endregion

#region Access

func set_batch_size(new_batch_size : int) -> void:
	if new_batch_size < 0:
		push_error("Batch size can't be negative")
		return
	if new_batch_size == 0:
		push_error("Batch size must be positive, zero isn't a positive number")
		return
	batch_size = new_batch_size

func set_input(input : Array) -> void:
	if input.size() != neurons_out[0].size():
		push_error("Input size must match number of neurons on the first layer")
		return
	neurons_out[0] = input.duplicate()

func set_target(new_target : Array) -> void:
	if target.size() != new_target.size():
		push_error("Target size must match number of neurons on the last layer")
		return
	target = new_target.duplicate()

func get_output() -> Array:
	return neurons_out[f.size() - 1].duplicate()

func last_layer() -> int:
	return structure.size() - 1

func get_total_weights() -> int:
	var weights_quantity : int = 0
	var i : int = 0
	while i < structure.size() - 1:
		weights_quantity += structure[i] * structure[i + 1]
		i += 1
	return weights_quantity

func get_total_biases() -> int:
	var biases_quantity : int = 0
	var i : int = 1
	while i < structure.size():
		biases_quantity += structure[i]
		i += 1
	return biases_quantity * int(use_bias)

func get_loss(input_data : Array, target_data : Array) -> float:
	if input_data.size() == 0:
		push_error("Provide input data")
		return -1.0
	if input_data.size() != target_data.size():
		push_error("Number of elements in input data array doesn't match number of elements in target data array")
		return -1.0
	var loss : float = 0
	var i : int = 0
	while i < input_data.size():
		set_input(input_data[i])
		propagate_forward()
		loss += lf.call(get_output(), target_data[i])
		i += 1
	return loss / input_data.size()

func assign(nn : NNET) -> void:
	aa = deep_copy(nn.aa)
	structure = deep_copy(nn.structure)
	weights = deep_copy(nn.weights)
	neurons_in = deep_copy(nn.neurons_in)
	neurons_out = deep_copy(nn.neurons_out)
	biases = deep_copy(nn.biases)
	neuron_deltas = deep_copy(nn.neuron_deltas)
	weight_deltas = deep_copy(nn.weight_deltas)
	bias_deltas = deep_copy(nn.bias_deltas)
	wapply_deltas = deep_copy(nn.wapply_deltas)
	bapply_deltas = deep_copy(nn.bapply_deltas)
	target = deep_copy(nn.target)
	uf = deep_copy(nn.uf)
	dlayer = deep_copy(nn.dlayer)
	lr = nn.lr
	use_bias = nn.use_bias
	algorithm = nn.algorithm
	batch_size = nn.batch_size
	copy_functions(nn)

func duplicate() -> NNET:
	var nn : NNET = NNET.new([1,1], false)
	nn.assign(self)
	return nn

func copy_functions(nn : NNET) -> void:
	f.resize(nn.f.size())
	df.resize(nn.f.size())
	var i : int = 0
	while i < nn.f.size():
		if fd[i] != BNNET.ActivationFunctions.user_function:
			set_function(nn.fd[i], i)
		else:
			set_function(nn.uf[i], i)
		i += 1
	fd[fd.size() - 1] = nn.fd[fd.size() - 1]
	lf = nn.lf

#endregion

#region Data

func save_binary(path : String) -> void:
	var file = FileAccess.open(full_path(path), FileAccess.WRITE)
	# ------------ checking for errors
	if not file.is_open():
		push_error("Could not open file: ", error_string(file.get_error()))
	# ------------ saving
	file.store_string("NNETB " + version + "\n") # NNETB 3.1.0
	file.store_32(structure.size())
	for layer in structure:
		file.store_64(layer)
	file.store_8(int(use_bias))
	for function in fd:
		file.store_8(function) # the last function is loss function.
	# ------------ saving weights
	var i : int = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				file.store_double(weights[i][j][k])
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				file.store_double(biases[i][j])
				j += 1
			i += 1
	# ------------

func save_nonbinary(path : String) -> void:
	var file = FileAccess.open(full_path(path), FileAccess.WRITE)
	# ------------ checking for errors
	if not file.is_open():
		push_error("Could not open file: ", error_string(file.get_error()))
	# ------------ saving important things
	file.store_line("NNETN " + version) # NNETB 3.1.0
	file.store_line("structure : " + str(structure).erase(0,1).left(-1).replace(",",""))
	# ------------ saving activation functions
	var i : int = 0
	while i < f.size():
		file.store_line("function layer " + str(i) + " : " + str(fd[i]) + " (" + activation_to_string(fd[i]) + ")")
		i += 1
	# ------------ saving loss function
	file.store_line("loss function : " + str(fd[f.size()]) + " (" + loss_to_string(fd[f.size()]) + ")")
	# ------------ saving other data
	file.store_line("weights : " + str(get_total_weights()))
	file.store_line("biases : " + str(get_total_biases() * int(use_bias)))
	# ------------ saving weights
	i = 0
	while i < weights.size():
		var j : int = 0
		while j < weights[i].size():
			var k : int = 0
			while k < weights[i][j].size():
				file.store_line("weight : " + str(weights[i][j][k]))
				k += 1
			j += 1
		i += 1
	if use_bias:
		i = 0
		while i < biases.size():
			var j : int = 0
			while j < biases[i].size():
				file.store_line("bias : " + str(biases[i][j]))
				j += 1
			i += 1
	# ------------ closing file
	file.close()
	# ------------

func load_data(path : String) -> void:
	var file = FileAccess.open(full_path(path), FileAccess.READ)
	# ------------ checking for errors
	if not file.is_open():
		push_error("Could not open file: ", error_string(file.get_error()))
	var file_string : String = file.get_buffer(11).get_string_from_utf8()
	if file_string != "NNETB 3.1.0" and file_string != "NNETN 3.1.0":
		# ------------ updating file 
		file.close()
		old_file_to_new(path)
		# ------------ loading
		load_data(path)
		return
		# ------------
	# ------------ loading file
	if file_string == "NNETB 3.1.0":
		# ------------ loading binary data
		file.seek(0)
		file.get_line()
		var new_structure : Array = []
		new_structure.resize(file.get_32())
		var i : int = 0
		while i < new_structure.size():
			new_structure[i] = file.get_64()
			i += 1
		self._init(new_structure, bool(file.get_8()))
		i = 0
		while i < structure.size():
			set_function(file.get_8(), i)
			i += 1
		set_loss_function(file.get_8())
		# ------------ loading weights
		i = 0
		while i < weights.size():
			var j : int = 0
			while j < weights[i].size():
				var k : int = 0
				while k < weights[i][j].size():
					weights[i][j][k] = file.get_double()
					k += 1
				j += 1
			i += 1
		if use_bias:
			i = 0
			while i < biases.size():
				var j : int = 0
				while j < biases[i].size():
					biases[i][j] = file.get_double()
					j += 1
				i += 1
		# ------------
	elif file_string == "NNETN 3.1.0":
		# ------------ loading non-binary data
		file.seek(0)
		file.get_buffer(12)
		var new_structure : Array = Array(file.get_line().get_slice(" : ", 1).split(" ")).map(func(string_neurons): return string_neurons.to_int())
		# ------------ skipping to biases
		var i : int = 0
		while i < new_structure.size() + 2:
			file.get_line()
			i += 1
		# ------------ initialising
		self._init(new_structure, file.get_line().get_slice(" : ", 1).to_int() > 0)
		# ------------ returning to functions
		file.seek(0)
		file.get_line()
		file.get_line()
		i = 0
		while i < structure.size():
			set_function(file.get_line().get_slice(" : ", 1).get_slice(" ", 0).to_int(), i)
			i += 1
		set_loss_function(file.get_line().get_slice(" : ", 1).get_slice(" ", 0).to_int())
		# ------------ skipping unnecessary data
		file.get_line()
		file.get_line()
		# ------------ loading weights
		i = 0
		while i < weights.size():
			var j : int = 0
			while j < weights[i].size():
				var k : int = 0
				while k < weights[i][j].size():
					weights[i][j][k] = file.get_line().get_slice(" : ", 1).to_float()
					k += 1
				j += 1
			i += 1
		if use_bias:
			i = 0
			while i < biases.size():
				var j : int = 0
				while j < biases[i].size():
					biases[i][j] = file.get_line().get_slice(" : ", 1).to_float()
					j += 1
				i += 1
		# ------------
	# ------------ closing file
	file.close()
	# ------------

#region Side functions

func activation_to_string(activation_function : BNNET.ActivationFunctions) -> String:
	match activation_function:
		BNNET.ActivationFunctions.identity:
			return "identity"
		BNNET.ActivationFunctions.binary_step:
			return "binary step"
		BNNET.ActivationFunctions.logistic:
			return "logistic"
		BNNET.ActivationFunctions.tanh:
			return "tanh"
		BNNET.ActivationFunctions.ReLU:
			return "ReLU"
		BNNET.ActivationFunctions.mish:
			return "mish"
		BNNET.ActivationFunctions.swish:
			return "swish"
		BNNET.ActivationFunctions.softmax:
			return "softmax"
		BNNET.ActivationFunctions.user_function:
			return "user function"
	push_error("Activation function does not exist")
	return "ERROR"

func loss_to_string(loss_function : BNNET.LossFunctions) -> String:
	match fd[f.size()]:
		BNNET.LossFunctions.MSE:
			return "MSE"
		BNNET.LossFunctions.MAE:
			return "MAE"
		BNNET.LossFunctions.BCE:
			return "BCE"
		BNNET.LossFunctions.CCE:
			return "CCE"
		BNNET.LossFunctions.Cosine_similarity_loss:
			return "Cosine similarity loss"
		BNNET.LossFunctions.LogCosh_loss:
			return "LogCosh loss"
		BNNET.LossFunctions.Hinge_loss:
			return "Hinge loss"
		BNNET.LossFunctions.user_function:
			return "User loss"
	push_error("Loss function does not exist")
	return "ERROR"

func algorithm_to_string(current_algoritm : BNNET.Algorithms) -> String:
	match current_algoritm:
		BNNET.Algorithms.gradient_descent:
			return "gradient descent"
		BNNET.Algorithms.adamW:
			return "Adam(W)"
		BNNET.Algorithms.adamax:
			return "Adamax"
		BNNET.Algorithms.adadelta:
			return "Adadelta"
		BNNET.Algorithms.yogi:
			return "Yogi"
		BNNET.Algorithms.NAG:
			return "NAG"
		BNNET.Algorithms.nadam:
			return "Nadam"
		BNNET.Algorithms.resilient_propagation:
			return "Rprop"
		BNNET.Algorithms.no_algorithm:
			return "No algorithm"
	push_error("Algorithm does not exist")
	return "ERROR"

#endregion

#endregion
