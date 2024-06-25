@tool
extends EditorPlugin

func example() -> void:
	var nn : NNET = NNET.new([2,5,6,1], false)
	nn.set_loss_function(BNNET.LossFunctions.MSE)
	nn.use_Rprop(0.3)
	#nn.use_gradient_descent(1.0)
	#nn.use_Adam(0.1)
	#nn.use_NAG(1.0,0.9)
	#nn.use_Nadam(0.1)
	#nn.use_Adamax(0.1)
	#nn.use_Yogi(0.1)
	#nn.use_Adadelta(0.9)
	nn.set_batch_size(4)
	
	XOR_test(nn)
	
	for i in range(1500) :
		nn.train(
		[[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]],
		[[0.0], [1.0], [1.0], [0.0]])
	
	XOR_test(nn)

func XOR_test(nn : NNET) -> void:
	print("----------------------------------------------------")
	print("Loss: ", nn.get_loss([[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]], [[0.0], [1.0], [1.0], [0.0]]))
	nn.set_input([0.0,0.0]); nn.propagate_forward()
	print("0, 0: ", nn.get_output())
	nn.set_input([1.0,0.0]); nn.propagate_forward()
	print("1, 0: ", nn.get_output())
	nn.set_input([0.0,1.0]); nn.propagate_forward()
	print("0, 1: ", nn.get_output())
	nn.set_input([1.0,1.0]); nn.propagate_forward()
	print("1, 1: ", nn.get_output())
	print("----------------------------------------------------")
