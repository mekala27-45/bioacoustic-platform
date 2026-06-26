"""Verifier for the c-inet task.

Compiles the agent's /app/src/inet.c exactly as the contract prescribes, then
differentials its output against Python's ipaddress module on a committed
adversarial corpus (golden.txt) plus a freshly generated random batch. The agent
cannot read this file or the golden, and the fresh batch is computed live from
ipaddress, so neither hardcoding the corpus nor reading the answers is possible.
"""
import ipaddress as ip
import pathlib
import random
import subprocess

SRC = pathlib.Path("/app/src/inet.c")
BIN = "/tmp/inet_built"
GOLDEN = pathlib.Path(__file__).parent / "golden.txt"


def _compile():
    assert SRC.is_file(), "missing agent source at /app/src/inet.c"
    r = subprocess.run(
        ["gcc", "-O2", "-std=c11", "-o", BIN, str(SRC)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, "agent source failed to compile:\n" + r.stderr[:2000]


_BUILT = False


def _ensure_built():
    global _BUILT
    if not _BUILT:
        _compile()
        _BUILT = True


def _run(inputs):
    _ensure_built()
    data = "".join(s + "\n" for s in inputs)
    p = subprocess.run([BIN], input=data, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, f"binary exited {p.returncode}: {p.stderr[:500]}"
    out = p.stdout.split("\n")
    if out and out[-1] == "":
        out = out[:-1]
    assert len(out) == len(inputs), f"got {len(out)} output lines for {len(inputs)} inputs"
    return out


def _ref(s):
    # Out of scope: ipv4-mapped/embedded (both '.' and ':') and zone ids ('%').
    a = s.split("/", 1)[0]
    if "%" in s or ("." in a and ":" in a):
        return "INVALID"
    try:
        if "/" in s:
            return ip.ip_network(s, strict=False).compressed
        return ip.ip_address(s).compressed
    except ValueError:
        return "INVALID"


def _category(inp, exp):
    if exp == "INVALID":
        return "invalid"
    a = inp.split("/", 1)[0]
    cidr = "/" in inp
    v6 = ":" in a
    if cidr:
        return "cidr_v6" if v6 else "cidr_v4"
    return "ipv6" if v6 else "ipv4"


def _load_golden():
    ins, exps = [], []
    for ln in GOLDEN.read_text(encoding="utf-8").splitlines():
        i, e = ln.split("\t", 1)
        ins.append(i)
        exps.append(e)
    return ins, exps


_GINS, _GEXPS = _load_golden()
_GGOT = None


def _golden_got():
    global _GGOT
    if _GGOT is None:
        _GGOT = _run(_GINS)
    return _GGOT


def _check_category(cat):
    got = _golden_got()
    bad = [(_GINS[i], _GEXPS[i], got[i]) for i in range(len(_GEXPS))
           if _category(_GINS[i], _GEXPS[i]) == cat and got[i] != _GEXPS[i]]
    assert not bad, f"{len(bad)} {cat} mismatches; first 8: {bad[:8]}"


def test_source_present_and_compiles():
    _ensure_built()
    assert pathlib.Path(BIN).is_file()


def test_ipv4():
    _check_category("ipv4")


def test_ipv6():
    _check_category("ipv6")


def test_cidr_v4():
    _check_category("cidr_v4")


def test_cidr_v6():
    _check_category("cidr_v6")


def test_invalid_rejected():
    _check_category("invalid")


def _gen_fresh(seed, n):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.4:
            out.append(".".join(str(rng.randint(0, 255)) for _ in range(4)))
        elif roll < 0.5:
            octs = [rng.randint(0, 255) for _ in range(4)]
            j = rng.randint(0, 3)
            octs[j] = rng.choice([256, 999, -1])
            out.append(".".join(str(o) for o in octs))
        elif roll < 0.85:
            groups = [rng.randint(0, 0xFFFF) for _ in range(8)]
            if rng.random() < 0.6:
                st = rng.randint(0, 6)
                for k in range(st, min(8, st + rng.randint(1, 3))):
                    groups[k] = 0
            # skip the ipv4-mapped range so canonicalization stays pure hex
            if groups[:5] == [0, 0, 0, 0, 0] and groups[5] == 0xFFFF:
                groups[5] = 0xFFFE
            out.append(":".join(format(g, "x") for g in groups))
        else:
            if rng.random() < 0.5:
                base = ".".join(str(rng.randint(0, 255)) for _ in range(4))
                out.append(base + "/" + str(rng.randint(0, 32)))
            else:
                groups = [rng.randint(0, 0xFFFF) for _ in range(8)]
                out.append(":".join(format(g, "x") for g in groups) + "/" + str(rng.randint(0, 128)))
    return out


def test_fresh_random_batch():
    """Anti-hardcode: a fresh seeded batch, scored live against ipaddress."""
    fresh = _gen_fresh(seed=99173, n=800)
    exps = [_ref(s) for s in fresh]
    got = _run(fresh)
    bad = [(fresh[i], exps[i], got[i]) for i in range(len(fresh)) if got[i] != exps[i]]
    assert not bad, f"{len(bad)} fresh mismatches; first 8: {bad[:8]}"
