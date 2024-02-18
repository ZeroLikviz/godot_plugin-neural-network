# NeuralNet Toolkit
<p align="center">
<img src="https://i.postimg.cc/9FWkc5GR/250-20240218012009.png" alt="NNET logotype" style="image-rendering: pixelated">
</p>
That is an asset for working with neural networks in Godot.

-------

### NNET
To start working with the NNET class, create a variable of the type NNET, here's an example:
```GDScript
...
var neural_network : NNET = NNET.new(...)
```
Initialization of \*NNET classes requires some parameters:
- structure, it may look like that: \[4,13,7,2\] (see <font style="color:red">structure</font>)
- learning rate
- true or false depending on whether you are going to use biases or not
- <font style="color:grey">( Optionally )</font> true or false depending on whether you are going to use f'() or not  (see <font style="color:red">true_fd</font>)
And now we have something similar to it:
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
`[0.48292014681138]`

`[0.49121472848005]`

`[0.512]`
After this little test, you may want to use a bit more of the functionality the NNET class provides, like activation functions, saving and loading neural networks, etc.

All the other NNET functions, that you may want to use:
- set_function (see <font style="color:red">enum ActivationFunction</font>)
- set_custom_function (see <font style="color:red">set_custom_function</font>)
- duplicate (see <font style="color:red">duplicate</font>)
- assign (see <font style="color:red">assign</font>)
- save_data (see <font style="color:red">save_data</font>)
- load_data (see <font style="color:red">load_data</font>)
- copy_from_file (see <font style="color:red">copy_from_file</font>)
- print_info (see <font style="color:red">print_info</font>)


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
 
After you are done training, you may want to save your agent's brains, and you can accomplish this by getting a neural network that is responsible for the agent's actions (see <font style="color:red">get_main</font>) and then invoking the save_data method (see <font style="color:red">save_data</font>)

When you are changing the initial environment, for example, changing from level 0 to 1, or moving a wall (doing any actions that the agent is not responsible for), you should call the reset function (see <font style="color:red">reset</font>). Don't be afraid of its name; it doesn't mean the neural network of the agent will be cleared or something; by calling this function, you are only telling the agent that the initial environment has been changed.

All the other RLNNET functions you may want to use:
- get_main_output (see <font style="color:red">get_main_output</font>)

### GPUNNET
This class is very similar to NNET, but there are some limitations and disadvantages:
- after you are done with the variable of type GPUNNET, you must call the free_objects function.
- there are no save/load data functions for the GPUNNET class. (I will add them by the 24-25th of February)
- the allocated memory for the neurons is a rectangular 2D array, that means if you create a neural network with the structure like this \[1,10,50,5,1\], then 2D array with the size of 5 times 50 will be allocated for the neurons. That is 194 inactive neurons, so structure like \[1,50,50,50,1\] would be better for the memory usage.


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







