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
	NN.set_function(NNET.ActivationFunction.ReLU)
	NN.set_desired_output([0.5])
	NN.set_input([0.1])
	NN.run()
	NN.print_output()
	NN.train(500)
	var h = NN.get_output()[0]
	if not is_equal_approx(h, 0.5):
		print_rich("[color=red]ERROR[/color] " + str(h))
		return
	NN.print_output()

func _ready() -> void:
	print("Here, the neural network tries to make the output equal to 0.5. The training method used is back propagation.")
	basic_test()
	print("Here, the neural network tries to make the output equal to 0.0. The training method used is reinforcement learning.")
	rl_test()
	print()
	#perfomance_test()
	#components_test()
	#gpunnet_components_test()

func components_test() -> void:
	print("testing save/load systems: ")
	var NN = NNET.new([1,1])
	NN.set_function(NNET.ActivationFunction.linear)
	NN.set_input([1.0])
	NN.run()
	var output : float = NN.get_output()[0]
	print_rich("    [color=magenta]saving data[/color]")
	NN.save_data("test data.txt")
	print_rich("    [color=magenta]loading data[/color]")
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
	print_rich("    [color=magenta]copying data[/color]")
	NN_container.copy_from_file("test data.txt")
	NN_container.set_input([1.0])
	NN_container.run()
	if NN_container.get_output()[0] == output:
		print_rich("    [color=green]test was passed[/color]")
	else: print_rich("    [color=red]test was failed[/color] : [color=yellow]" + str(output) + " != " + str(NN_container.get_output()[0]) + "[/color]")

func perfomance_test() -> void:
	var cpu_nn_small : NNET    =    NNET.new([5,5])
	var gpu_nn_small : GPUNNET = GPUNNET.new([5,5])
	
	print("Structure: [5, 5]. CPU NN run function time: ", time_us(func()->void:cpu_nn_small.run()), " us")
	print("Structure: [5, 5]. GPU NN run function time: ", time_us(func()->void:gpu_nn_small.run()), " us")
	
	var cpu_nn : NNET    =    NNET.new([5,128,128,5])
	var gpu_nn : GPUNNET = GPUNNET.new([5,128,128,5])
	
	print("Structure: [5, 128, 128, 5]. CPU NN run function time: ", time_us(func()->void:cpu_nn.run()), " us")
	print("Structure: [5, 128, 128, 5]. GPU NN run function time: ", time_us(func()->void:gpu_nn.run()), " us")
	
	var cpu_nn_large : NNET    =    NNET.new([5,512,512,128,5])
	var gpu_nn_large : GPUNNET = GPUNNET.new([5,512,512,128,5])
	
	print("Structure: [5, 512, 512, 128, 5]. CPU NN run function time: ", time_us(func()->void:cpu_nn_large.run()), " us")
	print("Structure: [5, 512, 512, 128, 5]. GPU NN run function time: ", time_us(func()->void:gpu_nn_large.run()), " us")
	
	var cpu_nn_huge : NNET    =    NNET.new([5,1024,1024,1024,5])
	var gpu_nn_huge : GPUNNET = GPUNNET.new([5,1024,1024,1024,5])
	
	print("Structure: [5, 1024, 1024, 1024, 5]. CPU NN run function time: ", time_us(func()->void:cpu_nn_huge.run()), " us")
	print("Structure: [5, 1024, 1024, 1024, 5]. GPU NN run function time: ", time_us(func()->void:gpu_nn_huge.run()), " us")
	
	gpu_nn_large.free_objects()
	gpu_nn_small.free_objects()
	gpu_nn_huge.free_objects()
	gpu_nn.free_objects()

func time_us(function : Callable) -> int:
	var time_start = Time.get_ticks_usec()
	function.call()
	var time_end   = Time.get_ticks_usec()
	return time_end - time_start

func gpunnet_components_test():
	print("testing GPUNNET class :")
	print_rich("[color=magenta]    initializing GPUNNET network[/color]")
	var gp : GPUNNET = GPUNNET.new([5,2], 1000.0, false)
	print_rich("[color=magenta]    setting the input for GPUNNET network[/color]")
	gp.set_input([0.0,0.0,0.0,0.0,0.0])
	print_rich("[color=magenta]    invoking the GPUNNET run() method[/color]")
	gp.run()
	if not is_equal_approx(gp.get_output()[0], 0.5):
		print_rich("    [color=red]test was failed[/color] : [color=magenta]",str(gp.get_output()[0]), " != 0.5","[/color]")
	else: print_rich("    [color=green]test was passed[/color]")
	gp.set_input([0.0,1.0,1.0,0.0,0.0])
	print_rich("[color=magenta]    setting the desired output[/color]")
	gp.set_desired_output([500.0,500.0])
	print_rich("[color=magenta]    invoking GPUNNET train() method[/color]")
	gp.train(50)
	gp.run()
	if not is_equal_approx(gp.get_output()[0], 1.0):
		print_rich("    [color=red]test was failed[/color] : [color=magenta]",str(gp.get_output()[0]), " != 1.0","[/color]")
	else: print_rich("    [color=green]test was passed[/color]")
	gp.free_objects()
