#!/usr/bin/env python3
"""Behaviour-cloning trainer: a dotbot-bc v1 export in, a dotbot-mlp v1 net (.dotmlp + .f32) out.

Plain numpy, no Godot and no PyTorch: the net is at most two hidden layers of 64 tanh units, which trains in seconds on a CPU.

    python3 train.py <export dir> --out <dir> [--name NAME] [--epochs N] [--seed N] [--hidden 64,64]
                     [--dagger file.py:function --dagger-rounds N]
    python3 train.py --selftest

Inputs are divided by the manifest's scale, as DotBotMlp does at run time. A continuous action field is regressed on value/scale and its scale is folded into the last layer's weights, so the net outputs raw action values like the rows hold; a discrete (button) field is a logit trained with binary cross entropy, so readers apply a sigmoid.

DAgger hook: --dagger names `file.py:function`; after each round the function is called as `function(policy, round)` with `policy(obs_raw[n, obs]) -> action_raw[n, act]` (discrete fields as probabilities) and returns None or `(obs_raw, action_raw)` rows labelled by the expert on states the policy visited. They are appended to the training set and training continues.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile

import numpy as np

EXPORT_FORMAT = "dotbot-bc"
EXPORT_VERSION = 1
MLP_FORMAT = "dotbot-mlp"
MLP_VERSION = 1
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_WEIGHT_BYTES = 4 * 1024 * 1024
MAX_HIDDEN_LAYERS = 2
MAX_HIDDEN_WIDTH = 64
SHARD_RE = re.compile(r"^shard_[0-9]{1,6}\.f32$")


class ExportError(Exception):
    pass


# ---------------------------------------------------------------- export reading

def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_export(directory):
    """Returns (manifest, {'train': rows, 'held_out': rows}); rows are raw float32 [n, frame_size]. The directory is untrusted: checked."""
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        raise ExportError("no manifest.json in %s" % directory)
    if os.path.getsize(path) > MAX_MANIFEST_BYTES:
        raise ExportError("manifest over %d bytes" % MAX_MANIFEST_BYTES)
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict) or m.get("format") != EXPORT_FORMAT:
        raise ExportError("not a %s export" % EXPORT_FORMAT)
    if m.get("version") != EXPORT_VERSION:
        raise ExportError("export version %r; this reads %d" % (m.get("version"), EXPORT_VERSION))
    frame = m.get("frame_size")
    if not isinstance(frame, int) or frame < 1:
        raise ExportError("bad frame_size")
    for key in ("kind", "schema_id"):
        if not isinstance(m.get(key), str) or not m[key]:
            raise ExportError("'%s' is missing" % key)
    for group in ("observation", "action"):
        fields = m.get(group)
        if not isinstance(fields, list) or not fields:
            raise ExportError("'%s' must list fields" % group)
        for fld in fields:
            sc = fld.get("scale")
            if not isinstance(fld.get("size"), int) or fld["size"] < 1 or isinstance(sc, bool) or not isinstance(sc, (int, float)) or not math.isfinite(sc) or sc <= 0:
                raise ExportError("a field of '%s' has a bad size or scale" % group)
    osz = sum(f["size"] for f in m["observation"])
    asz = sum(f["size"] for f in m["action"])
    if osz + asz != frame:
        raise ExportError("fields add up to %d, not frame_size %d" % (osz + asz, frame))
    out = {"train": [], "held_out": []}
    for sh in m.get("shards", []):
        name = sh.get("file", "")
        if not isinstance(name, str) or not SHARD_RE.match(name) or sh.get("split") not in out:
            raise ExportError("bad shard entry %r" % (name,))
        p = os.path.join(directory, name)
        want = int(sh["rows"]) * frame * 4
        if not os.path.isfile(p) or os.path.getsize(p) != want:
            raise ExportError("shard %s has the wrong size" % name)
        with open(p, "rb") as f:
            data = f.read()
        if _sha256(data) != str(sh.get("sha256", "")).lower():
            raise ExportError("shard %s does not match its hash" % name)
        rows = np.frombuffer(data, dtype="<f4").reshape(-1, frame)
        if not np.isfinite(rows).all():
            raise ExportError("shard %s holds a value that is not finite" % name)
        out[sh["split"]].append(rows)
    return m, {k: (np.concatenate(v).astype(np.float32) if v else np.zeros((0, frame), np.float32)) for k, v in out.items()}


def field_layout(manifest):
    """Per-element observation scales, action scales and the discrete mask."""
    obs_scales = np.concatenate([np.full(f["size"], f["scale"], np.float64) for f in manifest["observation"]])
    act_scales = np.concatenate([np.full(f["size"], f["scale"], np.float64) for f in manifest["action"]])
    discrete = np.concatenate([np.full(f["size"], bool(f.get("discrete", False))) for f in manifest["action"]])
    return obs_scales, act_scales, discrete


# ---------------------------------------------------------------- the net

class Net:
    """Layers of (w[out, in], b[out]) in float64 while training; saved as float32."""

    def __init__(self, sizes, activations, rng=None):
        self.sizes = list(sizes)
        self.activations = list(activations)
        self.w, self.b = [], []
        for i in range(len(sizes) - 1):
            if rng is None:
                self.w.append(np.zeros((sizes[i + 1], sizes[i])))
            else:
                lim = math.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
                self.w.append(rng.uniform(-lim, lim, (sizes[i + 1], sizes[i])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, x, keep=False):
        acts = [x]
        for i in range(len(self.w)):
            z = acts[-1] @ self.w[i].T + self.b[i]
            a = self.activations[i]
            acts.append(np.tanh(z) if a == "tanh" else (np.maximum(z, 0.0) if a == "relu" else z))
        return acts if keep else acts[-1]

    def backward(self, acts, grad_out):
        gw, gb = [None] * len(self.w), [None] * len(self.w)
        g = grad_out
        for i in reversed(range(len(self.w))):
            a = self.activations[i]
            if a == "tanh":
                g = g * (1.0 - acts[i + 1] ** 2)
            elif a == "relu":
                g = g * (acts[i + 1] > 0)
            gw[i] = g.T @ acts[i]
            gb[i] = g.sum(0)
            g = g @ self.w[i]
        return gw, gb


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def loss_and_grad(net, x, y, discrete):
    """Mean over rows of: squared error on continuous outputs plus binary cross entropy on discrete logits."""
    acts = net.forward(x, keep=True)
    out = acts[-1]
    n = x.shape[0]
    grad = np.empty_like(out)
    cont = ~discrete
    err = out[:, cont] - y[:, cont]
    grad[:, cont] = 2.0 * err / n
    z, t = out[:, discrete], y[:, discrete]
    grad[:, discrete] = (_sigmoid(z) - t) / n
    loss = float((err ** 2).sum() / n + (np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))).sum() / n)
    return loss, acts, grad


class Adam:
    def __init__(self, net, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in net.w + net.b]
        self.v = [np.zeros_like(p) for p in net.w + net.b]

    def step(self, net, gw, gb):
        self.t += 1
        for p, g, m, v in zip(net.w + net.b, gw + gb, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


def prepare(rows, manifest):
    obs_scales, act_scales, discrete = field_layout(manifest)
    n_obs = len(obs_scales)
    rows = np.asarray(rows)
    x = rows[:, :n_obs].astype(np.float64) / obs_scales
    y = rows[:, n_obs:].astype(np.float64) / act_scales
    y[:, discrete] = np.clip(y[:, discrete], 0.0, 1.0)
    return x, y


def evaluate(net, x, y, discrete):
    if x.shape[0] == 0:
        return {}
    out = net.forward(x)
    cont = ~discrete
    res = {}
    if cont.any():
        res["mse"] = float(np.mean((out[:, cont] - y[:, cont]) ** 2))
    if discrete.any():
        res["button_accuracy"] = float(np.mean((out[:, discrete] > 0) == (y[:, discrete] > 0.5)))
    return res


def train(manifest, train_rows, held_rows, hidden=(64, 64), epochs=60, batch=256, lr=3e-3, seed=1,
          dagger=None, dagger_rounds=0, log=print):
    if len(hidden) > MAX_HIDDEN_LAYERS or any(h < 1 or h > MAX_HIDDEN_WIDTH for h in hidden):
        raise ExportError("at most %d hidden layers of at most %d" % (MAX_HIDDEN_LAYERS, MAX_HIDDEN_WIDTH))
    obs_scales, act_scales, discrete = field_layout(manifest)
    rng = np.random.default_rng(seed)
    x, y = prepare(train_rows, manifest)
    hx, hy = prepare(held_rows, manifest)
    if x.shape[0] == 0:
        raise ExportError("no training rows")
    net = Net([len(obs_scales), *hidden, len(act_scales)], ["tanh"] * len(hidden) + ["linear"], rng)

    def policy(obs_raw):
        out = net.forward(np.asarray(obs_raw, np.float64) / obs_scales)
        return np.where(discrete, _sigmoid(out), out * act_scales)

    history = []
    for rnd in range(dagger_rounds + 1):
        opt = Adam(net, lr)
        for ep in range(epochs):
            order = rng.permutation(x.shape[0])
            total = 0.0
            for s in range(0, x.shape[0], batch):
                idx = order[s:s + batch]
                loss, acts, grad = loss_and_grad(net, x[idx], y[idx], discrete)
                gw, gb = net.backward(acts, grad)
                opt.step(net, gw, gb)
                total += loss * len(idx)
            if ep == epochs - 1 or ep % 20 == 0:
                log("round %d epoch %d train loss %.5f" % (rnd, ep, total / x.shape[0]))
        history.append(evaluate(net, hx, hy, discrete))
        if dagger is not None and rnd < dagger_rounds:
            extra = dagger(policy, rnd)
            if extra is not None and len(extra[0]):
                ex, ey = prepare(np.concatenate([np.asarray(extra[0], np.float64), np.asarray(extra[1], np.float64)], axis=1), manifest)
                x, y = np.concatenate([x, ex]), np.concatenate([y, ey])
                log("dagger round %d added %d rows" % (rnd, ex.shape[0]))
    metrics = {"train_rows": int(x.shape[0]), "held_out_rows": int(hx.shape[0]), "epochs": epochs, "seed": seed,
               "dagger_rounds": dagger_rounds, "train": evaluate(net, x, y, discrete), "held_out": history[-1]}
    return net, metrics


# ---------------------------------------------------------------- writing weights

def to_runtime_layers(net, manifest):
    """float32 layers with the continuous outputs' scale folded into the last layer, so the net outputs raw action values."""
    _, act_scales, discrete = field_layout(manifest)
    mult = np.where(discrete, 1.0, act_scales)
    layers = []
    for i in range(len(net.w)):
        w, b = net.w[i].copy(), net.b[i].copy()
        if i == len(net.w) - 1:
            w *= mult[:, None]
            b *= mult
        layers.append((w.astype(np.float32), b.astype(np.float32)))
    return layers


def write_net(out_dir, name, manifest, net, metrics):
    os.makedirs(out_dir, exist_ok=True)
    obs_scales, _, _ = field_layout(manifest)
    blob = b"".join(w.astype("<f4").tobytes() + b.astype("<f4").tobytes() for w, b in to_runtime_layers(net, manifest))
    if len(blob) > MAX_WEIGHT_BYTES:
        raise ExportError("weights over 4 MiB")
    header = {
        "format": MLP_FORMAT, "version": MLP_VERSION, "schema_id": manifest["schema_id"], "kind": manifest["kind"],
        "sizes": net.sizes, "activations": net.activations, "input_scales": [float(s) for s in obs_scales],
        "sha256": _sha256(blob), "metrics": metrics,
    }
    with open(os.path.join(out_dir, name + ".f32"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out_dir, name + ".dotmlp"), "w", encoding="utf-8") as f:
        json.dump(header, f, indent=1, sort_keys=True)
        f.write("\n")
    return header


def runtime_forward(header, blob, obs_raw):
    """What DotBotMlp computes, from the written files only (float32 math like the runtime)."""
    x = np.asarray(obs_raw, np.float32) / np.asarray(header["input_scales"], np.float32)
    sizes, off = header["sizes"], 0
    flat = np.frombuffer(blob, dtype="<f4")
    for i, act in enumerate(header["activations"]):
        n_in, n_out = sizes[i], sizes[i + 1]
        w = flat[off:off + n_in * n_out].reshape(n_out, n_in)
        b = flat[off + n_in * n_out:off + n_in * n_out + n_out]
        off += n_in * n_out + n_out
        z = x @ w.T + b
        x = np.tanh(z) if act == "tanh" else (np.maximum(z, 0) if act == "relu" else z)
    return x


# ---------------------------------------------------------------- synthetic export (tests)

def write_synthetic_export(directory, seed=7, rows=3000, held=500):
    """A dotbot-bc v1 export of a known function, laid out as DotBotDemoExport writes it (for tests without a recording)."""
    rng = np.random.default_rng(seed)
    obs_fields = [{"name": "position", "size": 3, "scale": 100.0, "discrete": False}, {"name": "velocity", "size": 3, "scale": 20.0, "discrete": False}]
    act_fields = [{"name": "yaw", "size": 1, "scale": 3.14159, "discrete": False}, {"name": "move", "size": 2, "scale": 1.0, "discrete": False}, {"name": "jump", "size": 1, "scale": 1.0, "discrete": True}]

    def make(n):
        obs = np.concatenate([rng.uniform(-100, 100, (n, 3)), rng.uniform(-20, 20, (n, 3))], axis=1)
        p, v = obs[:, :3] / 100.0, obs[:, 3:] / 20.0
        yaw = 3.14159 * np.tanh(p[:, 0] - v[:, 1])
        move = np.stack([np.tanh(v[:, 0] + p[:, 2]), np.sin(p[:, 1])], axis=1)
        jump = (p[:, 2] + v[:, 2] > 0.2).astype(np.float64)
        return np.concatenate([obs, yaw[:, None], move, jump[:, None]], axis=1).astype("<f4")

    os.makedirs(directory, exist_ok=True)
    shards = []
    for i, (split, n) in enumerate((("train", rows), ("held_out", held))):
        data = make(n).tobytes()
        name = "shard_%d.f32" % i
        with open(os.path.join(directory, name), "wb") as f:
            f.write(data)
        shards.append({"file": name, "split": split, "rows": n, "sha256": _sha256(data), "episodes": [[1 + i, n]], "map_id": "toy", "map_version": "v1"})
    manifest = {
        "format": EXPORT_FORMAT, "version": EXPORT_VERSION, "kind": "ride", "schema_version": 1, "schema_id": "synthetic0",
        "frame_size": 10, "observation_size": 6, "action_size": 4, "observation": obs_fields, "action": act_fields,
        "split": {"seed": "bc1", "held_out_fraction": 0.15}, "maps": [], "shards": shards, "held_out": [],
        "totals": {"train_rows": rows, "held_out_rows": held}, "rejected": [],
    }
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest


# ---------------------------------------------------------------- command line

def _load_hook(spec):
    path, _, fn = spec.rpartition(":")
    s = importlib.util.spec_from_file_location("dagger_hook", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return getattr(mod, fn)


def selftest():
    quiet = lambda s: None
    with tempfile.TemporaryDirectory() as tmp:
        exp, out = os.path.join(tmp, "export"), os.path.join(tmp, "out")
        write_synthetic_export(exp)
        manifest, sets = load_export(exp)
        obs_scales, act_scales, discrete = field_layout(manifest)
        cont = ~discrete

        def dagger(policy, rnd):
            # the "expert" is the synthetic function; label states the policy would visit
            rng = np.random.default_rng(100 + rnd)
            obs = np.concatenate([rng.uniform(-100, 100, (200, 3)), rng.uniform(-20, 20, (200, 3))], axis=1)
            p, v = obs[:, :3] / 100.0, obs[:, 3:] / 20.0
            act = np.stack([3.14159 * np.tanh(p[:, 0] - v[:, 1]), np.tanh(v[:, 0] + p[:, 2]), np.sin(p[:, 1]), (p[:, 2] + v[:, 2] > 0.2) * 1.0], axis=1)
            assert policy(obs).shape == act.shape
            return obs, act

        net, metrics = train(manifest, sets["train"], sets["held_out"], hidden=(32, 32), epochs=80, seed=3, dagger=dagger, dagger_rounds=1, log=quiet)
        assert metrics["train_rows"] == 3200, metrics
        header = write_net(out, "toy", manifest, net, metrics)
        blob = open(os.path.join(out, "toy.f32"), "rb").read()
        held = sets["held_out"]
        got = runtime_forward(header, blob, held[:, :6])
        assert np.mean((got[:, cont] - held[:, 6:][:, cont]) ** 2) < 0.05, "does not fit the synthetic function"
        assert metrics["held_out"]["button_accuracy"] > 0.85, metrics
        # the written net agrees with the in-memory one (float32 rounding only)
        mem = net.forward(held[:, :6].astype(np.float64) / obs_scales)
        assert np.allclose(got[:, cont], (mem * act_scales)[:, cont], atol=1e-3)
        assert np.allclose(got[:, discrete], mem[:, discrete], atol=1e-3)
        # a layer wider than 64 or a third hidden layer is refused
        for bad in ((65,), (8, 8, 8)):
            try:
                train(manifest, sets["train"], sets["held_out"], hidden=bad, epochs=1, log=quiet)
                raise AssertionError("accepted hidden %r" % (bad,))
            except ExportError:
                pass
        # a tampered shard is refused
        with open(os.path.join(exp, "shard_0.f32"), "r+b") as f:
            f.write(b"\x00\x00\x80\x3f")
        try:
            load_export(exp)
            raise AssertionError("tampered shard accepted")
        except ExportError:
            pass
    print("selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Behaviour-cloning trainer: dotbot-bc export in, .dotmlp and .f32 out.")
    ap.add_argument("export_dir", nargs="?")
    ap.add_argument("--out", default=".")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--dagger", default=None, help="file.py:function, see the module doc")
    ap.add_argument("--dagger-rounds", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.export_dir:
        ap.error("an export directory is required")
    try:
        manifest, sets = load_export(a.export_dir)
        hidden = tuple(int(h) for h in a.hidden.split(",") if h)
        net, metrics = train(manifest, sets["train"], sets["held_out"], hidden, a.epochs, a.batch, a.lr, a.seed,
                             _load_hook(a.dagger) if a.dagger else None, a.dagger_rounds)
        name = a.name or "%s_%s" % (manifest["kind"], manifest["schema_id"])
        write_net(a.out, name, manifest, net, metrics)
    except ExportError as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2
    print("wrote %s.dotmlp and %s.f32 in %s; held-out %s" % (name, name, a.out, json.dumps(metrics["held_out"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
