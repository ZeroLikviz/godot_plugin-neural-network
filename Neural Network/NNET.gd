@icon("res://addons/Neural Network/base NNET class.gd")
extends BaseNNET
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
var uf : Array[Callable] = []
var f : Array[Callable] = []
var df : Array[Callable] = []
var lr : float = 1.0 # learning rate
var use_bias : bool
var lf : Callable # loss function
var dlayer : Array[float] = []
var get_bias : Callable
var algorithm : BaseNNET.Algorithm = BaseNNET.Algorithm.no_algorithm
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

#endregion

#region Functions

func set_function(function : Variant, from : int, to : int = from) -> void:
	if from < 0:
		push_error("There is no negative layers, don't use negative numbers")
		return
	if to >= structure.size():
		push_error("You exceeded number of layers by ", to - structure.size() + 1 ,".Remember: From and To are both inclusive")
		return
	if function is Callable or function is BaseNNET.ActivationFunctions:
		var layer : int = from
		while layer <= to:
			specify_function(function, layer)
			layer += 1
	else:
		push_error("Function must be Callable or belong to BaseNNET.ActivationFunctions")

func specify_function(function : Variant, layer : int) -> void:
	if function is Callable:
		fd[layer] = BaseNNET.ActivationFunctions.user_function
		uf[layer] = function
		f[layer] = func() -> void:
			var i : int = 0
			while i < neurons_in[layer].size():
				neurons_in[layer][i] = uf[layer].call(neurons_in[layer][i])
				i += 1
		df[layer] = func() -> void:
			var i : int = 0
			while i < neurons_in[layer].size():
				dlayer[i] = (uf[layer].call(neurons_in[layer][i]) - uf[layer].call(neurons_in[layer][i] + apzero)) / apzero
				i += 1
	elif function is BaseNNET.ActivationFunctions:
		fd[layer] = function
		match function:
			BaseNNET.ActivationFunctions.identity:
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
			BaseNNET.ActivationFunctions.binary_step:
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
			BaseNNET.ActivationFunctions.logistic:
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
			BaseNNET.ActivationFunctions.tanh:
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
			BaseNNET.ActivationFunctions.ReLU:
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
			BaseNNET.ActivationFunctions.mish:
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
			BaseNNET.ActivationFunctions.swish:
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
			BaseNNET.ActivationFunctions.softmax:
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
		push_error("Function must be Callable or belong to BaseNNET.ActivationFunctions")

func set_loss_function(function : Variant) -> void:
	if function is Callable:
		fd[f.size()] = BaseNNET.LossFunctions.user_function
		lf = function
		# ------------- test
		var array : Array = []
		array.resize(f.size())
		array.fill(1.2345)
		var test_error : float = lf.call(array, array)
		var i : int = 0
		while i < f.size():
			if not is_equal_approx(array[i], 1.2345):
				push_error("It's not allowed for loss function to change output/target values. Change your function")
				i += 1
		# -------------
	elif function is BaseNNET.LossFunctions:
		fd[f.size()] = function
		match function:
			BaseNNET.LossFunctions.MSE:
				lf = func(outputs, targets) -> float:
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						sum += (outputs[i] - targets[i]) * (outputs[i] - targets[i])
						i += 1
					sum /= outputs.size()
					return sum
			BaseNNET.LossFunctions.MAE:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						var part_sum : float = outputs[i] - targets[i]
						sum += int(part_sum > 0) * part_sum + int(part_sum < 0) * -part_sum
						i += 1
					sum /= outputs.size()
					return sum
			BaseNNET.LossFunctions.BCE:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						sum += targets[i] * log(outputs[i]) + (1.0 - targets[i]) * log(1.0 - outputs[i])
						i += 1
					sum /= outputs.size()
					return -sum
			BaseNNET.LossFunctions.CCE:
				lf = func(outputs, targets):
					var i : int = 0
					var sum : float = 0.0
					while i < outputs.size():
						sum += targets[i] * log(outputs[i])
						i += 1
					sum /= outputs.size()
					return -sum
			BaseNNET.LossFunctions.Hinge_loss:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						var sum_part : float = 1.0 - targets[i] * outputs[i]
						sum += int(sum_part > 0) * sum_part
						i += 1
					sum /= outputs.size()
					return sum
			BaseNNET.LossFunctions.Cosine_similarity_loss:
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
			BaseNNET.LossFunctions.LogCosh_loss:
				lf = func(outputs, targets):
					var sum : float = 0.0
					var i : int = 0
					while i < outputs.size():
						sum += log(cosh(outputs[i] - targets[i]))
						i += 1
					sum /= outputs.size()
					return sum
	else:
		push_error("Function must be Callable or belong to BaseNNET.LossFunctions")

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
			neurons_in[layer][neuron] = get_bias.call(layer - 1, neuron)
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
	if use_bias:
		allocate2(biases, -1)
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
	set_function(BaseNNET.ActivationFunctions.logistic, 0, structure.size() - 1)
	init_callables()
	set_loss_function(BaseNNET.LossFunctions.MSE)

func reinit() -> void:
	fill_rand3(weights)
	if use_bias:
		fill_rand2(biases)
	match algorithm:
		BaseNNET.Algorithm.adamW:
			use_Adam(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BaseNNET.Algorithm.adamax:
			use_Adamax(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BaseNNET.Algorithm.nadam:
			use_Nadam(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BaseNNET.Algorithm.yogi:
			use_Yogi(lr, aa[Adam.beta1], aa[Adam.beta2], aa[Adam.weight_decay])
		BaseNNET.Algorithm.NAG:
			use_NAG(lr, aa[NAG.beta])
		BaseNNET.Algorithm.resilient_propagation:
			use_Rprop(aa[Rprop.uv], aa[Rprop.ep], aa[Rprop.em], aa[Rprop.max_step], aa[Rprop.min_step])
		BaseNNET.Algorithm.adadelta:
			use_Adadelta(aa[Adadelta.df])

func init_callables() -> void:
	if use_bias:
		get_bias = func(layer : int, neuron : int) -> float:
			return biases[layer][neuron]
	else:
		get_bias = func(_layer : int, _neuron : int) -> float:
			return 0.0

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

func use_gradient_decent(learning_rate : float) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	algorithm = BaseNNET.Algorithm.gradient_descent

func use_Adam(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_adam(beta_1, beta_2, weights_decay)
	algorithm = BaseNNET.Algorithm.adamW

func use_Nadam(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_adam(beta_1, beta_2, weights_decay)
	algorithm = BaseNNET.Algorithm.nadam

func use_Adamax(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) -> void:
	kill_additions()
	check_learning_rate(learning_rate)
	init_adam(beta_1, beta_2, weights_decay)
	algorithm = BaseNNET.Algorithm.nadam

func use_Yogi(learning_rate : float, beta_1 :