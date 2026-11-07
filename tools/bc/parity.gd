extends SceneTree
func _init() -> void:
	var dir := OS.get_environment("PARITY_DIR")
	var cases: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(dir + "/cases.json"))
	var res := DotBotMlp.load_file(dir + "/toy")
	if not res.ok:
		print("LOAD FAILED: ", res.message if "message" in res else res)
		quit(1)
		return
	var net: DotBotMlp = res.value
	print(net.describe())
	var worst := 0.0
	for i in cases["obs"].size():
		var o := PackedFloat32Array()
		for v in cases["obs"][i]:
			o.append(v)
		var out := net.forward(o)
		for j in out.size():
			worst = maxf(worst, absf(out[j] - float(cases["expected"][i][j])))
	print("PARITY max abs diff ", worst, " over ", cases["obs"].size(), " cases")
	quit(0 if worst < 1e-5 else 1)
