@tool
extends Node

func rl_test() -> void:
	var RLNN = RLNNET.new([1,3,2,1], 0.5, true)
	RLNN.set_input([0.18])
	RLNN.show_result()
	for i in range(500):
		# here I'm calculating reward for what neural network does. In this case the less output, the higher the reward, so neural network will try to minimize the output to maximize the reward
		RLNN.set_reward(1.0 - RLNN.get_output()[0])
		RLNN.update()
	RLNN.show_result()

func basic_test() -> void:
	var NN = NNET.new([1,1], 0.5, true)
	NN.set_desired_output([0.5])
	NN.set_input([0.1])
	NN.run()
	NN.show_result()
	NN.train(500)
	NN.show_result()

func _ready() -> void:
	print("Here, the neural network tries to make the output equal to 0.5. The training method used is back propagation.")
	basic_test()
	print("Here, the neural network tries to make the output equal to 0.0. The training method used is reinforcement learning.")
	rl_test()
	print()
	
	#components_test()

func components_test() -> void:
	print("testing save/load systems")
	var NN = NNET.new([1,1])
	NN.set_input([1.0])
	NN.run()
	var output : float = NN.get_output()[0]
	print("saving data")
	NN.save_data("test data.txt")
	print("loading data")
	var NN_container = NNET.new([1,1])
	NN_container.load_data("test data.txt")
	NN_container.set_input([1.0])
	NN_container.run()
	if NN_container.get_output()[0] == output:
		print("test was passed")
		return
	print("test was failed")
