@tool
extends Node

func rl_test() -> void:
	var RLNN = RLNNET.new([1,3,2,1], 0.5, true)
	RLNN.set_input([0.18])
	RLNN.print_output()
	for i in range(500):
		# here I'm calculating reward for what neural network does. In this case the less output, the higher the reward, so neural network will try to minimize the output to maximize the reward
		RLNN.set_reward(1.0 - RLNN.get_output()[0])
		RLNN.update()
	RLNN.print_output()

func basic_test() -> void:
	var NN = NNET.new([1,1], 0.5, true)
	NN.set_desired_output([0.5])
	NN.set_input([0.1])
	NN.run()
	NN.print_output()
	NN.train(500)
	NN.print_output()

func _ready() -> void:
	print("Here, the neural network tries to make the output equal to 0.5. The training method used is back propagation.")
	basic_test()
	print("Here, the neural network tries to make the output equal to 0.0. The training method used is reinforcement learning.")
	rl_test()
	print()
	
	#components_test()

func components_test() -> void:
	print("testing save/load systems: ")
	var NN = NNET.new([1,1])
	NN.set_input([1.0])
	NN.run()
	var output : float = NN.get_output()[0]
	print_rich("    [color=green]saving data[/color]")
	NN.save_data("test data.txt")
	print_rich("    [color=green]loading data[/color]")
	var NN_container = NNET.new([1,1])
	if NN_container.load_data("test data.txt") == -1:
		print_rich("    [color=red]test was failed[/color]")
	NN_container.set_input([1.0])
	NN_container.run()
	if NN_container.get_output()[0] == output:
		print_rich("    [color=green]test was passed[/color]")
	else: print_rich("    [color=red]test was failed[/color] : [color=yellow]" + str(output) + " != " + str(NN_container.get_output()[0]) + "[/color]")
	
	print()
	print("testing copy_from_file system :")
	NN.print_info("NN", 4)
	NN_container.print_info("NN_container", 4)
	print_rich("    [color=green]copying data[/color]")
	NN_container.copy_from_file("test data.txt")
	NN_container.set_input([1.0])
	NN_container.run()
	if NN_container.get_output()[0] == output:
		print_rich("    [color=green]test was passed[/color]")
	else: print_rich("    [color=red]test was failed[/color] : [color=yellow]" + str(output) + " != " + str(NN_container.get_output()[0]) + "[/color]")