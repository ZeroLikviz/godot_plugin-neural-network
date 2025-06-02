## Loss functions for evaluating neural network performance.
class_name LossFunctions

## Mean Squared Error (MSE): Computes the average squared difference between predicted and target values.
static var MSE: Callable = func(predicted: PackedFloat32Array, target: PackedFloat32Array) -> float:
	var sum: float = 0.0
	for i in range(predicted.size()):
		var diff: float = predicted[i] - target[i]
		sum += diff * diff
	return sum / predicted.size()

## Mean Absolute Error (MAE): Computes the average absolute difference between predicted and target values.
static var MAE: Callable = func(predicted: PackedFloat32Array, target: PackedFloat32Array) -> float:
	var sum: float = 0.0
	for i in range(predicted.size()):
		sum += abs(predicted[i] - target[i])
	return sum / predicted.size()

## Binary Cross-Entropy (BCE): Computes the loss for binary classification, assuming predictions are in [0, 1].
static var BCE: Callable = func(predicted: PackedFloat32Array, target: PackedFloat32Array) -> float:
	var sum: float = 0.0
	const EPSILON: float = 1e-10 # Prevent log(0)
	for i in range(predicted.size()):
		var p: float = clamp(predicted[i], EPSILON, 1.0 - EPSILON)
		var t: float = target[i]
		sum += -(t * log(p) + (1.0 - t) * log(1.0 - p))
	return sum / predicted.size()

## Categorical Cross-Entropy (CCE): Computes the loss for multi-class classification, assuming predictions are probabilities.
static var CCE: Callable = func(predicted: PackedFloat32Array, target: PackedFloat32Array) -> float:
	var sum: float = 0.0
	const EPSILON: float = 1e-10 # Prevent log(0)
	for i in range(predicted.size()):
		var p: float = clamp(predicted[i], EPSILON, 1.0 - EPSILON)
		var t: float = target[i]
		sum += -t * log(p)
	return sum / predicted.size()

## Hinge Loss: Computes the loss for binary classification (e.g., SVM), assuming target is {-1, 1}.
static var Hinge: Callable = func(predicted: PackedFloat32Array, target: PackedFloat32Array) -> float:
	var sum: float = 0.0
	for i in range(predicted.size()):
		sum += max(0.0, 1.0 - predicted[i] * target[i])
	return sum / predicted.size()

## Cosine Similarity Loss: Computes 1 - cosine similarity between predicted and target vectors.
static var CosineSimilarity: Callable = func(predicted: PackedFloat32Array, target: PackedFloat32Array) -> float:
	var dot_product: float = 0.0
	var norm_pred: float = 0.0
	var norm_target: float = 0.0
	for i in range(predicted.size()):
		dot_product += predicted[i] * target[i]
		norm_pred += predicted[i] * predicted[i]
		norm_target += target[i] * target[i]
	var norm: float = sqrt(norm_pred) * sqrt(norm_target)
	return 1.0 - (dot_product / max(1e-10, norm)) # Prevent division by zero

## Log-Cosh Loss: Computes the logarithm of the hyperbolic cosine of the prediction error.
static var LogCosh: Callable = func(predicted: PackedFloat32Array, target: PackedFloat32Array) -> float:
	var sum: float = 0.0
	for i in range(predicted.size()):
		var diff: float = predicted[i] - target[i]
		sum += log(cosh(diff))
	return sum / predicted.size()
