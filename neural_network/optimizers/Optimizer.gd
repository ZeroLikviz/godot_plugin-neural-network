# Base class for optimizers used in neural network training.
class_name Optimizer

# Initializes optimizer-specific data for the given neural layer.
func initialize(layer: LayerCPU) -> void:
	pass

# Updates the layer's weights and biases using computed gradients.
func apply_gradients(layer: LayerCPU) -> void:
	pass

# Creates a duplicate of the optimizer instance.
func duplicate() -> Optimizer:
	return Optimizer.new()

# Returns the name of the optimizer class.
func get_class_name() -> String:
	return "Optimizer"

# Retrieves the optimizer's hyperparameters as an array.
func get_hyperparameters() -> Array:
	return []

# Loads hyperparameters from an array.
func load_hyperparameters(hyperparameters: Array) -> void:
	pass
 
