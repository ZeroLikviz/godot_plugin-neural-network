class_name ExpressionFunction

var callable: Callable
var expression: String

func _init(prototype: String, expression: String) -> void:
	self.expression = expression
	var gd_file : GDScript = GDScript.new()
	gd_file.source_code = "static var function : Callable = " + prototype + expression.indent("\t")
	gd_file.reload(false)
	callable = gd_file.function
