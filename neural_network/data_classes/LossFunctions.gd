## Loss functions for evaluating neural network performance.
class_name LossFunctions

## Mean Squared Error (MSE): Computes the average squared difference between predicted and target values.
static var MSE: String = "var sum: float = 0.0\n" + \
						 "for i in range(predicted.size()):\n" + \
						 "\tvar diff: float = predicted[i] - target[i]\n" + \
						 "\tsum += diff * diff\n" + \
						 "return sum / predicted.size()"

## Mean Absolute Error (MAE): Computes the average absolute difference between predicted and target values.
static var MAE: String = "var sum: float = 0.0\n" + \
						 "for i in range(predicted.size()):\n" + \
						 "\tsum += abs(predicted[i] - target[i])\n" + \
						 "return sum / predicted.size()\n"

## Binary Cross-Entropy (BCE): Computes the loss for binary classification, assuming predictions are in [0, 1].
static var BCE: String = "var sum: float = 0.0\n" + \
						 "const EPSILON: float = 1e-10 # Prevent log(0)\n" + \
						 "for i in range(predicted.size()):\n" + \
						 "\tvar p: float = clamp(predicted[i], EPSILON, 1.0 - EPSILON)\n" + \
						 "\tvar t: float = target[i]\n" + \
						 "\tsum += -(t * log(p) + (1.0 - t) * log(1.0 - p))\n" + \
						 "return sum / predicted.size()"

## Categorical Cross-Entropy (CCE): Computes the loss for multi-class classification, assuming predictions are probabilities.
static var CCE: String = "var sum: float = 0.0\n" + \
						 "const EPSILON: float = 1e-10 # Prevent log(0)\n" + \
						 "for i in range(predicted.size()):\n" + \
						 "\tvar p: float = clamp(predicted[i], EPSILON, 1.0 - EPSILON)\n" + \
						 "\tvar t: float = target[i]\n" + \
						 "\tsum += -t * log(p)\n" + \
						 "return sum / predicted.size()"

## Hinge Loss: Computes the loss for binary classification (e.g., SVM), assuming target is {-1, 1}.
static var Hinge: String = "var sum: float = 0.0\n" + \
						   "for i in range(predicted.size()):\n" + \
						   "\tsum += max(0.0, 1.0 - predicted[i] * target[i])\n" + \
						   "return sum / predicted.size()"

## Cosine Similarity Loss: Computes 1 - cosine similarity between predicted and target vectors.
static var CosineSimilarity: String = "var dot_product: float = 0.0\n" + \
									  "var norm_pred: float = 0.0\n" + \
									  "var norm_target: float = 0.0\n" + \
									  "for i in range(predicted.size()):\n" + \
									  "\tdot_product += predicted[i] * target[i]\n" + \
									  "\tnorm_pred += predicted[i] * predicted[i]\n" + \
									  "\tnorm_target += target[i] * target[i]\n" + \
									  "var norm: float = sqrt(norm_pred) * sqrt(norm_target)\n" + \
									  "return 1.0 - (dot_product / max(1e-10, norm)) # Prevent division by zero"

## Log-Cosh Loss: Computes the logarithm of the hyperbolic cosine of the prediction error.
static var LogCosh: String = "var sum: float = 0.0\n" + \
							 "for i in range(predicted.size()):\n" + \
							 "\tvar diff: float = predicted[i] - target[i]\n" + \
							 "\tsum += log(cosh(diff))\n" + \
							 "return sum / predicted.size()"
