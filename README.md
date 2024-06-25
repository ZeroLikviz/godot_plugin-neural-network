# Introduction 
<div style="text-align:center; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
  <img src="https://i.postimg.cc/7689k1gx/NNET-logo.png" alt="NNET logo" />
</div>

**NNET is an add-on which provides you functionality to work with neural networks.**

Any suggestions, ideas, improvements are welcome. If you want to share them, you can do so by submitting Github issue. Also if you find a bug please report it.

---
# NNET
NNET is a class that contains logic for creating, training, saving, and using neural networks. In this class, all computations are CPU-based.

### functions:
   **Init:**
   - **new**( architecture : Array, use_bias : bool ) - architecture (or structure) is an array that must contain at least two positive integers, representing the number of neurons in each layer. Use_bias is a variable that decides whether to use bias neurons or not. By default all layers functions are set to logistic activation function.
   - **reinit**( ) - initialises weights with random values, and resets algorithm's collected data, if algorithm has it.

**Set:**
   - **set_function**(function: Variant, layer_from: int, layer_to: int) - sets functions on layers from layer_from to layer_to, both inclusive. The first layer's index is zero; to get the last layer, use last_layer(). The function must either be Callable or belong to BNNET.ActivationFunctions (enums are in the next section, see ActivationFunctions). If the function is Callable, it must accept one parameter of type float and return a value of type float.
   - **set_loss_function**(function: Variant) - sets the loss function. The function must be either callable or belong to BNNET.LossFunctions (see the next section for enums). If the function is callable, it must accept two parameters: outputs and targets (predictions and ground truth or similar) and return a float value.
   - **set_batch_size**( size : int ) - sets batch size.
   - **set_input**( input : Array ) - sets input. If the number of elements in the input array does not match the number of neurons in the first layer, then an error is thrown.
   - **set_target**( target : Array ) - sets target. If the number of elements in the target array does not match the number of neurons in the last layer, then an error is generated.


**Get:**
   - **get_output**( ) - returns output.
   - **get_total_weights**( ) - returns number of weights (not including biases).
   - **get_total_biases**() - return number of biases.
   - **get_loss**( input_data : Array\[Array], target_data : Array\[Array] ) - returns loss for provided data. XOR example:
```GDScript
print("loss: ", nn.get_loss([[0,0], [0,1], [1,0], [1,1]] 
                            [[0],   [1],   [1]    [0]]))
```

**General:**
   - **duplicate**( ) - return a copy of neural network.
   - **assign**( nn : NNET ) - assigns the values and parameters of the provided neural network to the current neural network. Doesn't make data shared.
   - **last_layer**( ) - returns the last layer's index.
   - **propagate_forward**( ) - performs the forward propagation, computing outputs.

**Data:**
   - **save_binary**( path : String ) - saves structure, weights, and functions (except for user-made ones) in binary form. The path can either be a file name or a full path.
   - **save_nonbinary**( path : String ) - saves structure, weights, and functions (except for user-made ones) in a non-binary form. The path can either be a file name or a full path.
   - **load_data**( path : String ) - loads data to the neural network. All save files above version 2.0.0 can be used, though I recommend to make a copy of your save files, just in case.

**Use:**
   - **use_gradient_descent**( lr : float ) - selects the gradient descent algorithm as the optimisation algorithm.
   - **use_Rprop**( update_value : float = 0.1, eta_plus : float = 1.2, eta_minus : float = 0.5, maximum_step : float = 50.0, minimum_step = 0.000001 ) - selects the Rprop algorithm as the optimisation algorithm.
   - **use_Adam**( lr : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0 ) - selects the Adam algorithm as the optimisation algorithm. It actually uses AdamW, but if weights decay equals to zero, they are the same.
   - **use_NAG**( learning_rate : float, beta : float ) - selects the NAG algorithm as the optimisation algorithm.
   - **use_Adadelta**( damping_factor : float = 0.9 ) - selects the Adadelta algorithm as the optimisation algorithm.
   - **use_Yogi**( learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0 ) - selects the Yogi algorithm as the optimisation algorithm.
   - **use_Nadam**(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) - selects the Nadam algorithm as the optimisation algorithm.
   - **use_Adamax**(learning_rate : float, beta_1 : float = 0.9, beta_2 : float = 0.999, weights_decay : float = 0.0) - Selects the Adamax algorithm as the optimisation algorithm

> [!NOTE]
>    All variations of the Adam algorithm use weight decay as in the AdamW algorithm in my implementation

**Train:**
   - **train**( input_data : Array\[Array], target_data : Array\[Array] ) - optimises weights of the neural network using choosed algorithm. If the batch size is greater than number of elements in input/output data array, then an error is generated.

##### Code example of using NNET to solve XOR:
```GDScript
func _ready() -> void:
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
	
	for i in range(1000) :
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
	
```


### Enums

**ActivationFunctions:**
   - identity
   - binary_step
   - logistic / sigmoid / soft_step - by default, all layer functions are set to the logistic activation function.
   - tanh
   - ReLU
   - mish
   - swish
   - softmax
   - user_function - setting layer's function to the BNNET.ActivationFunctions.user_function doesn't do anything.

**LossFunctions:**
   - MSE - used by default
   - MAE
   - BCE
   - CCE - by cross entropy, usually categorical cross entropy is meant.
   - Hinge_loss
   - Cosine_similarity_loss
   - LogCosh_loss
   - user_function

**Algorithms:**
   - resilient_propagation
   - adamW
   - nadam
   - NAG
   - yogi
   - adamax
   - adadelta
   - gradient_descent
   - no_algorithm
>[!NOTE]
> You don't need this enum. It's used inside the NNET class only to remember which algorithm to use.

---

#### Plans:
- Try to learn PPO again to create the PPONNET or RLNNET class.
- Try to recreate the GPUNNET class. It might be cancelled, because it's quite difficult to control things on the GPU, and I have no Idea how to implement so many algorithms, without creating tons of .glsl files.
