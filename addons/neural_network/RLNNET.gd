@icon("res://addons/neural_network/RLNNET icon.png")
class_name RLNNET

var main : NNET = NNET.new([1,1],1.0,false)
var buffer : NNET = NNET.new([1,1],1.0,false)
var reward : float = 0.0
var reward_buffer : float = 0.0
var has_updated : bool = false
var has_runned : bool = false
var curiosity_rate : float = 0.1

func _init(layers_construction: Array = [1,1], curiosity_rate_a: float = 0.25, use_bias: bool = true, afd_a : bool = false) -> void:
	main.assign(NNET.new(layers_construction, 0.1, use_bias, afd_a))
	buffer.assign(main.duplicate())
	curiosity_rate = curiosity_rate_a

func mutate() -> void:
	has_runned = false
	var i : int = 0
	var size : int = buffer.weights.size()
	while i < size:
		buffer.weights[i] += randf_range(-curiosity_rate, curiosity_rate)
		i += 1
	for layer in range(buffer.biases.size()):
		i = 0
		while i < buffer.biases[layer].size():
			buffer.biases[layer][i] += randf_range(-curiosity_rate, curiosity_rate)
			i += 1

func set_input(input : Array[float]) -> void:
	main.set_input(input.duplicate(true))
	buffer.set_input(input.duplicate(true))
	has_runned = false

func run() -> void:
	if has_updated:
		buffer.run()
	else:
		main.run()
	has_runned = true

func get_output(transform : bool = false) -> Array:
	if not has_runned:
		run()
	if has_updated:
		return buffer.get_output(transform)
	return main.get_output(transform)

func set_function(function : NNET.ActivationFunction) -> void:
	main.set_function(function)
	buffer.set_function(function)

func set_custom_function(function : Callable) -> void:
	main.set_custom_function(function)
	buffer.set_custom_function(function)

func get_main_output(transform : bool = false) -> Array:
	main.run()
	return main.get_output(transform)

func set_reward(value : float) -> void:
	if not has_updated:
		reward = value
		return
	reward_buffer = value

func update() -> void:
	has_updated = true
	if reward_buffer > reward:
		main.assign(buffer)
		reward = reward_buffer
		reward_buffer = 0.0
		mutate()
		return
	reward_buffer = 0.0
	buffer.assign(main)
	mutate()

func reset() -> void:
	has_updated = false
	has_runned = false
	reward = 0.0
	reward_buffer = -1.0

func print_output() -> void:
	if not has_runned:
		run()
	if has_updated:
		buffer.print_output()
		return
	main.print_output()

func get_main() -> NNET:
	return main.duplicate()

func set_curiosity_rate(rate : float) -> void:
	curiosity_rate = rate
