## class used for training basic neural networks using reinforcement learning
##
## provides functional to train neural network using reinforcement learning. [br]
## to do that you need only 4 functions: [br]
## - [method RLNNET.set_reward] [br]
## - [method RLNNET.update] [br]
## - [method RLNNET.get_output] [br]
## - [method RLNNET.set_input] [br]
## [br]
## [param set_input()] using it you giving neural_network current state of enviroment [br]
## [param get_output()] it lets you to get what neural_network thinks [br]
## [param set_reward()] using it you say how good or bad current approach was [br]
## [param update()] it uses the reward to change current approach and make a better one. After calling this function reset the enviroment [br]

@icon("res://addons/neural_network/RLNNET icon.png")
class_name RLNNET

## contains main neural_network, you can think about it like main or stable approach
var main : NNET = NNET.new([1,1],1.0,false)
## contains extra neural_network, which used for testing new approaches
var buffer : NNET = NNET.new([1,1],1.0,false)
## reward
var reward : float = 0.0
## extra container for reward
var reward_buffer : float = 0.0
## indicates if updated function has been called
var has_updated : bool = false
## used for avoiding unnecessary runs, basically it is used for optimization
var has_runned : bool = false
## curiosity rate, essentially it is how much different new approaches will be from main approach. I recommend to use small values like 0.005
var curiosity_rate : float = 0.1
func _init(layers_construction: Array = [1,1], curiosity_rate_a: float = 0.008, use_bias: bool = true, range_a: NNET.RangeN = NNET.RangeN.R_0_1, afd_a : bool = false) -> void:
	main.assign(NNET.new(layers_construction, 0.1, use_bias, range_a, afd_a))
	main.enable_avoid_computations_mode()
	buffer.assign(main.duplicate())
	buffer.enable_avoid_computations_mode()

## creates new approach. You shouldn't use this function, forget about its existence. Using it without necessary knowledges can break everything down
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

## sets input
func set_input(input : Array[float]) -> void:
	main.set_input(input.duplicate(true))
	buffer.set_input(input.duplicate(true))
	has_runned = false

## runs neural network
func run() -> void:
	if has_updated:
		buffer.run()
	else:
		main.run()
	has_runned = true

## returns output of new approach. If new approach doesn't exist, it will return output of main approach
func get_output() -> Array:
	if not has_runned:
		run()
	if has_updated:
		return buffer.get_output()
	return main.get_output()

## generates output using main approach [member RLNNET.main]
func get_main_output() -> Array:
	main.run()
	return main.get_output()

## sets reward for current approach. Reward is used to calculate if new approach was better or worse
func set_reward(value : float) -> void:
	if not has_updated:
		reward = value
		return
	reward_buffer = value

## decides if new approach was worse or better, and creates new one. [br]
## you should provide reward to [method RLNNER.set_reward] before calling this function
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

## use it when you change initial environment
func reset() -> void:
	has_updated = false
	has_runned = false
	reward = 0.0
	reward_buffer = -1.0

## shows result of new approach. If new approach doesn't exist, it will return result of main approach
func show_result() -> void:
	if not has_runned:
		run()
	if has_updated:
		buffer.show_result()
		return
	main.show_result()

## returns copy of [member RLNNET.main]
func get_main() -> NNET:
	return main.duplicate()
## sets [member RLNNET.curiosity_rate]
func set_curiosity_rate(rate : float) -> void:
	curiosity_rate = rate