# Introduction 
<div style="text-align:center; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; image-rendering: pixelated;">
  <img src="https://i.postimg.cc/7689k1gx/NNET-logo.png" alt="NNET logo" />
</div>

**NNET is an add-on which provides you functionality for working with neural networks.**

Any suggestions, ideas, improvements are welcome. If you want to share them, you can do so by submitting Github issue. I'll try my best implementing your ideas, although I don't promise anything, if I can't add them, then I won't add them. Also if you find a bug please report it.

---
# NNET

NNET is a class, which contains logic for creating, training, saving and using neural networks. In this class all computations are CPU based.

##### creating neural network 
```GDScript
var nn : NNET = NNET.new([1,1], 0.1, false)
# here I created a neural network with two layers: input layer, which contains 1 neuron, and output layer, which contains 1 neuron too. I set learning rate to 0.1 and decided not to use bias neurons.
```

Function **new** accepts 3 parameters:
- structure   -   define it with an array. Structure of your neural network must contain at least two layers! example: `new([2,4,6,1]...`
- learning rate
- use bias   -   use true or false to say whether you want to use biases or not.

Now that you have created your NN, you can keep adjusting it. Here'are functions for it:
- **use_backpropagation**( learning_rate )   -   makes train function use backpropagation.
- **use_resilient_propagation**( update_value = 0.0125, multiplication_factor = 1.2, reduction_factor = 0.3, max_step = 1.0, min_step = 0.000001 )   -   makes train function use resilient propagation. This function has multiple parameters to adjust, though adjusting them all might be a little
tedious, so they all have standard values.
- **use_Adam**( learning_rate, beta_1 = 0.9, beta_2 = 0.999, new_weight_decay = 0.0 )   -   makes train function use Adam.
- **enable_dropout**( probability )   -   enables dropout, which is used to prevent NN from overfitting. Don't forget to disable dropout after training/before testing.
- **disable_dropout**() - disables dropout.
- **set_function**( function, layer )   -   *function* can be either Callable(x : float) -> float or belong to BaseNNET.ActivationFunctions. Here they're:
    - identity
	- binary_step
	- logistic or sigmoid or soft_step
	- tanh
	- ReLU
	- mish
	- swish
	- softmax
- **set_loss_function**( function )   -   *function* can be either Callable(outputs : Array, targets : Array) -> float or belong to BaseNNET.LossFunctions. Here they're:
	- MSE
	- MAE
	- BCE
	- CCE   -   don't forget to set activation function of the last layer to softmax when using Categorical Crossentropy
	- Hinge_loss
	- Cosine_similarity_loss
	- LogCosh_loss 

##### training neural network
```GDScript
...

for i in range(500):
    nn.set_input(input_data)
    nn.set_target(target_data) # previously it was set_desired_output()
    nn.train()
```

List of functions for training process:
- **set_input**( input )   -   sets input. If you have the same input data all the time, you don't need to set it twice. The input data remains until a new one is set.
- **set_target**( target )   -   in 1.x and 2.x versions it was called set_desired_output(). If you have the same target data all the time, you don't need to set it twice. The target data remains until a new one is set.
- **train**()   -   runs an iteration of training process, for which you can use backpropagation or resilient propagation. Second one is much faster, you can read about advantages and disadvantages of resilient propagation on the internet, if you want.

##### testing/using neural network
```GDScript
...

nn.save_data("file_name") # saving data

var nn2 : NNET = NNET.new([1,1], 0.1, false)
nn2.load_data("file_name") # loading data
# I've got a copy of nn, however I need to set again input data and activation functions.

nn.set_input(input_data)
nn.run()
nn.print_output()
var output = nn.get_output()
var logits = nn.get_logits()
```
List of useful functions and functions for testing:
- **run**()   -   makes NN to predict output. Don't forget to disable dropout, if you had enabled it.
- **get_output**()
- **get_logits**()   -   you can get raw data of neurons from the last layer through this function.
- **print_output**()   -   as the name suggests this function prints output.
- **BaseNNET.old_file_to_new**( path )   -   with the update format of save files have changed, so to make them useable again apply this function.
> [!WARNING]
> This function hasn't been fully tested, make a copy of your data files before using it.
- **save_data**( path, binary = true )   -   save weights and structure of your neural network with this function. With the second parameter you can choose between binary and non-binary formats.
- **load_data**( path )   -   loads weights and structure into your neural network. Activation functions, loss functions and all other stuff that is not weights or structure need to be specified again.
- **duplicate**()   -   in future.
- **assign**( NN )   -   in future.

---

**Planned for the future:**
- PPONNET
- GPUNNET*

GPUNNET might be **cancelled** because I had a lot of issues trying to make NN learn something. Furthermore, GPUNNET will not work on Android at all when running outside the editor. 
