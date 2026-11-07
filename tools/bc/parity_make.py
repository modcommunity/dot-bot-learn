"""Writes a toy net and a cases.json for parity.gd. Usage: parity_make.py <out dir>"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, train as t
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
t.write_synthetic_export(out + "/export")
m, sets = t.load_export(out + "/export")
net, met = t.train(m, sets["train"], sets["held_out"], hidden=(64, 64), epochs=40, seed=5, log=print)
h = t.write_net(out, "toy", m, net, met)
blob = open(out + "/toy.f32", "rb").read()
rng = np.random.default_rng(9)
obs = np.concatenate([rng.uniform(-100, 100, (200, 3)), rng.uniform(-20, 20, (200, 3))], axis=1).astype(np.float32)
# float64 reference of the float32 weights on the float32 inputs
x = obs.astype(np.float64) / np.array(h["input_scales"]); flat = np.frombuffer(blob, "<f4").astype(np.float64); off = 0
for i, a in enumerate(h["activations"]):
    ni, no = h["sizes"][i], h["sizes"][i+1]
    w = flat[off:off+ni*no].reshape(no, ni); b = flat[off+ni*no:off+ni*no+no]; off += ni*no+no
    x = x @ w.T + b
    if a == "tanh": x = np.tanh(x)
f32 = t.runtime_forward(h, blob, obs)
print("f64 vs trainer float32 forward max diff", float(np.abs(x - f32).max()))
json.dump({"obs": obs.tolist(), "expected": x.tolist(), "schema_id": h["schema_id"]}, open(out + "/cases.json", "w"))
