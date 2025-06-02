## A class which purpose is to simply store data
class_name NetworkConstants

static var EPS : float = 8e-3
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
	"func(x: float) -> float: return max(0.0, x)" : "func(x: float) -> float: return 1.0 if x > 0.0 else 0.0",
	"func(x: float) -> float: return x" : "func(x: float) -> float: return 1.0",
	"func(x: float) -> float: return 1.0 if x >= 0.0 else 0.0" : "func(x: float) -> float: return 0.0",
	"func(x: float) -> float: return 1.0 / (1.0 + exp(-x))" : "func(x: float) -> float: var s = 1.0 / (1.0 + exp(-x))\nreturn s * (1.0 - s)",
	"func(x: float) -> float: return tanh(x)" : "func(x: float) -> float: var t = tanh(x)\nreturn 1.0 - t * t",
	"func(x: float) -> float: return x * tanh(log(1.0 + exp(x)))" : "func(x: float) -> float: var u = log(1.0 + exp(x))\nvar t = tanh(u)\nreturn t + x * (1.0 - t * t) * (1.0 / (1.0 + exp(-x)))",
	"func(x: float) -> float: return x * (1.0 / (1.0 + exp(-x)))" : "func(x: float) -> float: var s = 1.0 / (1.0 + exp(-x))\nreturn s + x * s * (1.0 - s)"
}
