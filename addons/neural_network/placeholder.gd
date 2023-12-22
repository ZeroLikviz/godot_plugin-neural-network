@tool
extends EditorPlugin

func _enter_tree() -> void:
	if not DirAccess.dir_exists_absolute("res://addons/neural_network/data/"):
		DirAccess.make_dir_absolute("res://addons/neural_network/data")
