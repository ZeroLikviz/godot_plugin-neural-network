## Provides activation functions and their derivatives for neural network layers.
class_name ActivationFunctions

## Rectified Linear Unit (ReLU): Returns x if x > 0, else 0.
static var ReLU: String = "func(x: float) -> float: return max(0.0, x)"

## Identity: Returns the input value unchanged (f(x) = x).
static var Identity: String = "func(x: float) -> float: return x"

## Binary Step: Returns 1 if x >= 0, else 0.
static var BinaryStep: String = "func(x: float) -> float: return 1.0 if x >= 0.0 else 0.0"

## Logistic (Sigmoid): Maps input to (0, 1) range using 1 / (1 + exp(-x)).
static var Logistic: String = "func(x: float) -> float: return 1.0 / (1.0 + exp(-x))"

## Sigmoid: Alias for Logistic, maps input to (0, 1) range.
static var Sigmoid: String = Logistic

## SoftStep: Alias for Logistic, maps input to (0, 1) range.
static var SoftStep: String = Logistic

## Hyperbolic Tangent (tanh): Maps input to (-1, 1) range.
static var Tanh: String = "func(x: float) -> float: return tanh(x)"

## Mish: Smooth activation function, x * tanh(ln(1 + exp(x))).
static var Mish: String = "func(x: float) -> float: return x * tanh(log(1.0 + exp(x)))"

## Swish (SiLU): Smooth activation function, x * sigmoid(x).
static var Swish: String = "func(x: float) -> float: return x * (1.0 / (1.0 + exp(-x)))"
