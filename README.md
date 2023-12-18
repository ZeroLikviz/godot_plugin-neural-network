The add-on provides 2 classes: NNET and RLNNET. NNET stands for NeuralNet Toolkit, while RLNNET stands for Reinforcement Learning NeuralNet Toolkit.

The distinction between NNET and RLNNET is as follows: NNET employs back propagation for training, whereas RLNNET utilizes reinforcement learning.

Usage of NNET
  Neural network initialization
   The initial step requires you to provide the architecture of your neural network. Firstly, specify the number of input neurons, followed by the sequence of hidden layers, and conclude with the number of output layers in the format [1,6,5,1].

   The second step entails providing the learning rate for the neural network.

   In the third step, you are required to indicate "true" or "false" depending on whether you intend to use bias neurons or not.

   By the fourth step, it is optional to provide a range for the neural network. If no range is specified, the default range will be from 0 to 1.

Here is an example of how it may appear in your code:

```GDScript
  func _ready() -> void:
	  var neural_network_1 = NNET.new([1,6,5,1], 0.1, true)
	  var neural_network_2 = NNET.new([5,20,30,40,15], 0.005, true)
	  var neural_nrtwork_3 = NNET.new([2,1], 0.5, true)
```

  Training
   To initiate the training of your neural network, it is imperative to supply input data. Accomplish this by employing the command "set_input()".

   Subsequently, you will need to establish the desired output you aim to obtain. This can be accomplished by utilizing the command "set_desired_output()".

   In the present context, to facilitate the learning process of the neural network, employ the "train()" function. "train()" function accepts an optional parameter that determines the number of training iterations. If this parameter is not specified, it defaults to 1.

  Using
   To acquire the output of the neural network, firstly execute the "run()" command, followed by the utilization of the "get_output()" function. If you solely desire to display the output in the console, utilize the "show_result()" command.

```GDScript
func _ready() -> void:
	  var neural_network = NNET.new([1,3,1], 0.1, true)
	  
	  neural_network.set_input([0.5])
	  neural_network.set_desired_output([0.333])
	  
	  neural_network.run()
	  neural_network.show_result()
	  
	  neural_network.train()
	  print(neural_network.get_output())
	  
	  neural_network.train(500)
	  neural_network.show_result()
```
