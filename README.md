DOCUMENTATION IS USELESS. I will create new documentation about 18-19-th of February.
- 
TODO:
- ~~add user friendly functions to GPUNNET class~~
- create new documentation and learn how to create good documentations

documentation is in progress...

# NeuralNet Toolkit
<p align="center">
<img src="https://i.postimg.cc/9FWkc5GR/250-20240218012009.png" alt="NNET logotype">
</p>

-------
## User guide for working with NNET, RLNNET, GPUNNET classes in your project



## NNET
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

The neural network's structure looks like the image below if you define its structure as same as mine
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

All other NNET functions, that you may want to use:
- set_function (see <font style="color:red">enum ActivationFunction</font>)
- set_custom_function (see <font style="color:red">set_custom_function</font>)
- duplicate (see <font style="color:red">duplicate</font>)
- assign (see <font style="color:red">assign</font>)
- save_data (see <font style="color:red">save_data</font>)
- load_data (see <font style="color:red">load_data</font>)
- copy_from_file (see <font style="color:red">copy_from_file</font>)
- print_info (see <font style="color:red">print_info</font>)


	 