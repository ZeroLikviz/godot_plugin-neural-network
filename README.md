# Neural Networks plugin for Godot
<p align="center">
<img src="https://i.postimg.cc/9FWkc5GR/250-20240218012009.png" alt="NNET logotype" style="image-rendering: pixelated">
</p>
That is a plugin for working with neural networks in Godot.

-------

### NNET
To start working with the NNET class, create a variable of the type NNET, here's an example:
```GDScript
...
var neural_network : NNET = NNET.new(...)
```
[Initialization](#^nnetinit) of \*NNET classes requires some parameters:
- structure, it may look like that: \[4,13,7,2\]
- learning rate
- true or false depending on whether you are going to use biases or not
- <font style="color:grey">( Optionally )</font> true or false depending on whether you are going to use f'() or not  (see <font style="color:red">true_fd</font>)
 
And now we have something similar to this:
```GDScript
var neural_network : NNET = NNET.new([1,5,1], 1.0, true)
```

The neural network's structure looks like the image below if you define its structure as same as mine. (biases are not shown in the image)
![](https://i.postimg.cc/yN0pWDJP/249-20240217224512.png)
 
Neural networks need input data, so provide some. You can do it using the set_input method
```GDScript
neural_network.set_input([0.2])
```
Now let's run it and see what result we get
```GDScript
neural_network.run()
```
To get results, you can use the get_output function, but if you just want to print them, you can use the print_output method.
```GDScript
neural_network.print_output()
```
Usually the output you get the first time is not what you want, so you may want to train your neural network.
Before you start training, think about what you want to get. You can set your desired output with the set_desired_output method.
```GDScript
neural_network.set_desired_output([0.512])
```
Now you can start the training itself using train function.
```GDScript
neural_network.train()
neural_network.run()
```
Let's look at the result again.
```GDScript
neural_network.print_output()
```
The results are very likely closer to what you want, but still not good enough. So let's invoke the train function again, but this time with one extra parameter, which determines how many iterations the neural network will be training.
```GDScript
neural_network.train(250)
neural_network.run()
```
Now the result should be perfect or close to it. If it isn't, then repeat training process.
```GDScript
neural_network.print_output()
```
The output will be something like this:

```
[0.48292014681138]
[0.49121472848005]
[0.512]
```
After this little test, you may want to use a bit more of the functionality the NNET class provides, like activation functions, saving and loading neural networks, etc.

All the other NNET functions, that you may want to use:
- set_function
- set_custom_function
- duplicate
- assign
- save_data
- load_data
- copy_from_file
- print_info


### RLNNET
( I'm not even sure if this is reinforcement training. All these lectures/lessons/videos have confused me too much, and I would be glad if you could tell if this is RL or not )
 
Initialization process is the same, but instead of learning rate you provide "curiosity rate" (or mutation rate).

Before training the agent you need to:
- initialize the enviroment
- initialize the agent

To train agent follow the instructions:
- provide input data to agent
- get the actions the agent wants to take and then execute them
- check if the environment says to stop (game ended or agent died and etc.)
- in case if environment did say to stop:
	- provide reward to agent.
	- update the agent.
	- reset the enviroment.
- start this loop from the beginning and repeat until you are satisfied with the result
Notice that you should always provide the reward to the agent before calling the update function.

Architecture of your code may look similar to this:
```GDScript
var agent : RLNNET
func _ready() -> void:
	# initialazing the environment...
	agent = RLNNET.new(structure, curiosity_rate, use_bias)

func execute_agent_actions() -> void:
	# executing agent actions...
	pass


func main_loop() -> void:
	agent.set_input(inputs)
	execute_agent_actions(agent.get_output())
	if game_ended: # (or agent_died. Interpret it as you want)
		agent.set_reward(reward)
		agent.update()
		reset_enviroment()
```
Notice that it is not necessary to call the run function before invoking the get_output method. The reason is that the RLNNET class works a bit differently, and in case you didn't run the neural network before getting the output, the class will do it for you.
 
After you are done training, you may want to save your agent's brains, and you can accomplish this by getting a neural network that is responsible for the agent's actions and then invoking the save_data method

When you are changing the initial environment, for example, changing from level 0 to 1, or moving a wall (doing any actions that the agent is not responsible for), you should call the reset function. Don't be afraid of its name; it doesn't mean the neural network of the agent will be cleared or something; by calling this function, you are only telling the agent that the initial environment has been changed.

All the other RLNNET functions you may want to use:
- get_main_output

### GPUNNET
This class is very similar to NNET, but there are some limitations and disadvantages:
- after you are done with the variable of type GPUNNET, you must call the free_objects function.
- there are no save/load data functions for the GPUNNET class. (I will add them by the 24-25th of February)
- the allocated memory for the neurons is a rectangular 2D array, that means if you create a neural network with the structure like this \[1,10,50,5,1\], then 2D array with the size of 5 times 50 will be allocated for the neurons. That is 193 inactive neurons, so structure like \[1,50,50,50,1\] would be better for the memory usage.


Beside a few disadvantages, this class has one very powerful benefit: speed!
CPU will be faster on a small neural networks, but in case if your neural network is large (\[5,128,128,128,5\] for example), then GPU will be way faster. Here's characteristics from my device:
```
Structure: [5, 5]. CPU NN run function time: 150 us
Structure: [5, 5]. GPU NN run function time: 7277 us
Structure: [5, 128, 128, 5]. CPU NN run function time: 40522 us
Structure: [5, 128, 128, 5]. GPU NN run function time: 13896 us
Structure: [5, 512, 512, 128, 5]. CPU NN run function time: 652749 us
Structure: [5, 512, 512, 128, 5]. GPU NN run function time: 16088 us
Structure: [5, 1024, 1024, 1024, 5]. CPU NN run function time: 4534535 us
Structure: [5, 1024, 1024, 1024, 5]. GPU NN run function time: 30807 us
```
 
If you want to test your own device, then you could use the function performance_test from the script "example.gd".


------


##### NNET enum ActivationFunction
- linear
- sigmoid
- logistic = sigmoid
- ReLU
- custom

#####  NNET methods
Description:
- \_init
	
	parameters:
	- structure : layer[int] = [1,1]
	- learning_rate : float = 1.0
	- use_biases : bool = true
	- true_fd : bool = false
	
	That is an function for initializing
- set_function -> void
	
	parameters:
	- function : NNET.ActivationFunction
	
	Use it to choose one of the activation functions. By default activation function is a sigmoid (logistic) function.
- set_custom_function -> void
	
	parameters:
	- function : Callable
	
	Use it to set custom activation function. Warning: Custom activation functions can't be saved in a file, so when you are getting your neural network back from the file, you must set the activation function to custom manually.
- set_desired_output -> void
	
	parameters:
	- desired_output : Array\[float]
	
	Use it to set desired output. It will throw an error if the size of your desired output doesn't match the size of the output.
- set_input -> void
	
	parameters:
	- input : Array\[float]
	
	Use it to set input. It will throw an error if the size of the input provided by you doesn't match the size of the actual input.
- run -> void
	
	No parameters.
	
	Use it to execute neural network.
- train -> void
	
	parameters:
	- iterations
	
	Use it to train neural network. Parameter "iterations" represents how many times the neural network will be training.
- get_output -> Array
	
	parameters:
	- transform : bool = false
	
	Use it to get output. Provide true if you want your output to range from -1 to 1 instead of 0 to 1 (this only works when the activation function is a sigmoid (logistic) function ).
- print_output -> void
	
	parameters:
	- transform : bool = false
	
	Use it to print output. It's literally `print(get_output(transform))` inside.
- duplicate -> NNET
	
	No parameters.
	
	Use it to get a duplicate of the neural network.
- assign -> void
	
	parameters:
	- neural_network
	
	Use it to assign one neural network to another. The data of provided neural network is being copied, so they don't have any shared memory between them.
- save_data -> void
	
	parameters:
	- file_name : String
	
	Use to save your neural network. If file_name starts with "res://" or "user://" then it's treated as a full path, otherwise it's treated as a file name and path for file calculated as "res://addons/neural_network/data/" + file_name. Custom functions can't be saved into a file.
- load_data -> int
	
	parameters:
	- file_name : String
	
	Use it to load data from a file to your neural network, but be careful. If the neural network's structure inside the file doesn't match the structure of your neural network, then the function will print an error and return ERR_INVALID_DATA, and no changes will be applied to your neural network. If their structures match, then the data will be successfully loaded into your neural network, and the function will return OK.
	The path is calculated exactly the same way as in the save_data function.
	Custom functions can't be loaded into a neural network.
- copy_from_file -> void
	
	parameters:
	- file_name : String
	
	Use it to copy neural network from file entirely. The path is calculated exactly the same way as in the save_data function. Custom functions can't be loaded into a neural network.
	Warning: be careful, using this function you can change the construction of your neural network.
- print_info -> void
	
	parameters:
	- neural_network_name : String
	- spaces : int = 0
	- print_weights : bool = false
	
	Use it to print information about your neural network. Neural_network_name is what it is; "spaces" is a parameter that says how many spaces should be before every printed line of information. "print_weights" is a parameter that indicates whether weights should be printed or not. Example:
```GDScript
var neural_network : NNET = NNET.new([1,2,3,4,8,2,1], 0.81, false)
neural_network.set_function( NNET.ActivationFunction.linear )
neural_network.print_info()
```
Output:
```
SomeName :
    structure:           [1, 2, 3, 4, 8, 2, 1]
    bias using:          false
    learning rate:       0.81
    true f'():           false
    activation function: linear
```
 

##### RLNNET methods
Description:
- \_init
	
	parameters:
	- structure : layer[int] = [1,1]
	- curiousity_rate : float = 0.25
	- use_biases : bool = true
	- true_fd : bool = false
	
	That is a function for initializing
- set_input -> void same description as in the NNET class
- run -> void same description as in the NNET class
- get_output -> Array
	
	Similar to the NNET.get_output function with the difference that if the run function hasn't been called, then it will invoke the run function for you.
- set_function -> void same description as in the NNET class
- set_custom_function -> void same description as in the NNET class
- print_output -> void same description as in the NNET class
- set_reward -> void
	
	parameters:
	- reward : float
	
	Use this function to set the reward. This function should always be called before calling the update function.
- update -> void
	
	No parameters.
	
	Use this function to indicate that game ended, agent died, and etc. You should always call the set_reward function before calling this one.
- get_main -> void
	
	No parameters.
	
	Use this function to get neural network that is responsible for the most best agent's approach.
- get_main_output -> void
	
	No parameters.
	
	Use this function to get output from the best agent's approach. You mustn't use the run function before calling this one, since it is just a waste of power ( the run function may use new approaches instead of the best one )


##### GPUNNET methods
Description:
- \_init same description as in the NNET class
- set_input -> void same description as in the NNET class
- set_desired_output -> void same description as in the NNET class
- run -> void same description as in the NNET class
- get_output -> Array
	
	No parameters.
	
	same description as in the NNET class, except for the transform parameter.
- print_output -> void
	
	No parameters.
	
	same description as in the NNET class, except for the transform parameter.
- free_objects
	
	No parameters.
	
	Use this function to free the GPUNNET variable. This function should always be called after you finish working with the neural network.


--------



TODO:
- add save/load data functions to the GPUNNET class
- make more optimizations

 
If you find any mistakes and misunderstandings in this documentation please report through github issues.
