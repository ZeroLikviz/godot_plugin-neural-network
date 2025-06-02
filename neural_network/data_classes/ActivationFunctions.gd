class_name ActivationFunctions

static var ReLU: String = "return max(0.0, x)"

static var Identity: String = "return x"

static var BinaryStep: String = "return 1.0 if x >= 0.0 else 0.0"

static var Logistic: String = "return 1.0 / (1.0 + exp(-x))"

static var Sigmoid: String = Logistic

static var SoftStep: String = Logistic

static var Tanh: String = "return tanh(x)"

static var Mish: String = "return x * tanh(log(1.0 + exp(x)))"

static var Swish: String = "return x * (1.0 / (1.0 + exp(-x)))"
