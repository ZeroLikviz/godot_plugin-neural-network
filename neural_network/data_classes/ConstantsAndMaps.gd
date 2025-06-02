## A class which purpose is to simply store data
class_name NetworkConstants

static var EPS : float = 1e-5
static var OPTIMIZERS_MAP : Dictionary = {
	"AdadeltaOptimizer": AdadeltaOptimizer,
	"AdamaxOptimizer": AdamaxOptimizer,
	"AdamOptimizer": AdamOptimizer,
	"GradientDescentOptimizer": GradientDescentOptimizer,
	"NadamOptimizer": NadamOptimizer,
	"NAGOptimizer": NAGOptimizer,
	"RpropOptimizer": RpropOptimizer,
	"YogiOptimizer": YogiOptimizer,
	"Optimizer": Optimizer
}
static var ACTIVATION_TO_DERIVATIVE : Dictionary = {
	"return max(0.0, x)" : "return 1.0 if x > 0.0 else 0.0",
	"return x" : "return 1.0",
	"return 1.0 if x >= 0.0 else 0.0" : "return 0.0",
	"return 1.0 / (1.0 + exp(-x))" : "var s = 1.0 / (1.0 + exp(-x))\nreturn s * (1.0 - s)",
	"return tanh(x)" : "var t = tanh(x)\nreturn 1.0 - t * t",
	"return x * tanh(log(1.0 + exp(x)))" : "var u = log(1.0 + exp(x))\nvar t = tanh(u)\nreturn t + x * (1.0 - t * t) * (1.0 / (1.0 + exp(-x)))",
	"return x * (1.0 / (1.0 + exp(-x)))" : "var s = 1.0 / (1.0 + exp(-x))\nreturn s + x * s * (1.0 - s)"
}
