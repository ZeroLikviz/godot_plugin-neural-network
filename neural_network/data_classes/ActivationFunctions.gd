## Provides activation functions and their derivatives for neural network layers.
class_name ActivationFunctions

## Rectified Linear Unit (ReLU): Returns x if x > 0, else 0.
static var ReLU: ActivationFunction = ActivationFunction.new(
		func(x: float) -> float: return max(0.0, x),
		func(x: float) -> float: return 1.0 if x > 0.0 else 0.0)

## Identity: Returns the input value unchanged (f(x) = x).
static var Identity: ActivationFunction = ActivationFunction.new(
		func(x: float) -> float: return x,
		func(x: float) -> float: return 1.0)

## Binary Step: Returns 1 if x >= 0, else 0.
static var BinaryStep: ActivationFunction = ActivationFunction.new(
		func(x: float) -> float: return 1.0 if x >= 0.0 else 0.0,
		func(x: float) -> float: return 0.0)

## Logistic (Sigmoid): Maps input to (0, 1) range using 1 / (1 + exp(-x)).
static var Logistic: ActivationFunction = ActivationFunction.new(
		func(x: float) -> float: return 1.0 / (1.0 + exp(-x)),
		func(x: float) -> float: 
			var sigmoid: float = 1.0 / (1.0 + exp(-x))
			return sigmoid * (1.0 - sigmoid))

## Sigmoid: Alias for Logistic, maps input to (0, 1) range.
static var Sigmoid: ActivationFunction = Logistic

## SoftStep: Alias for Logistic, maps input to (0, 1) range.
static var SoftStep: ActivationFunction = Logistic

## Hyperbolic Tangent (tanh): Maps input to (-1, 1) range.
static var Tanh: ActivationFunction = ActivationFunction.new(
		func(x: float) -> float: return tanh(x),
		func(x: float) -> float: 
			var tanh_x: float = tanh(x)
			return 1.0 - tanh_x * tanh_x)

## Mish: Smooth activation function, x * tanh(ln(1 + exp(x))).
static var Mish: ActivationFunction = ActivationFunction.new(
		func(x: float) -> float: return x * tanh(log(1.0 + exp(x))),
		func(x: float) -> float:
			var exp_x: float = exp(x)
			var softplus: float = log(1.0 + exp_x)
			var tanh_softplus: float = tanh(softplus)
			var sigmoid_x: float = exp_x / (1.0 + exp_x)
			return tanh_softplus + x * (1.0 - tanh_softplus * tanh_softplus) * sigmoid_x)

## Swish (SiLU): Smooth activation function, x * sigmoid(x).
static var Swish: ActivationFunction = ActivationFunction.new(
		func(x: float) -> float: return x * (1.0 / (1.0 + exp(-x))),
		func(x: float) -> float:
			var sigmoid: float = 1.0 / (1.0 + exp(-x))
			return sigmoid + x * sigmoid * (1.0 - sigmoid))

## Class to store an activation function and its derivative.
class ActivationFunction:
	## The activation function.
	var activation_function: Callable
	
	## The derivative of the activation function.
	var derivative_function: Callable
	
	## Initializes the activation function and its derivative.
	## [param activation] The activation function as a Callable.
	## [param derivative] The derivative function as a Callable.
	func _init(activation: Callable, derivative: Callable) -> void:
		activation_function = activation
		derivative_function = derivative
