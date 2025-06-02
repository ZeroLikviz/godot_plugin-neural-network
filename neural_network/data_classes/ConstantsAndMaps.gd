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
