#!/usr/bin/env python3
"""Generate the c-inet golden corpus from Python's ipaddress reference.

Each line of golden.txt is "<input>\t<expected>" where <expected> is the
canonical form (RFC 5952 for IPv6, dotted for IPv4, host-zeroed network for
CIDR) or the literal INVALID. The verifier feeds <input> to the agent binary
and requires byte-identical <expected> output.

This script is build-time tooling kept for provenance; golden.txt is committed.
"""
import ipaddress as ip
import random

random.seed(20260626)

R = []  # list of input strings


def canon(s: str) -> str:
    # CIDR if it contains a slash, else a bare address.
    try:
        if "/" in s:
            return ip.ip_network(s, strict=False).compressed
        return ip.ip_address(s).compressed
    except ValueError:
        return "INVALID"


def is_v4mapped(groups):
    # Exclude ::ffff:0:0/96 so canonicalization stays pure-hex (Python renders
    # that range back to dotted-quad, which is intentionally out of scope here).
    return groups[:5] == [0, 0, 0, 0, 0] and groups[5] == 0xFFFF


def rand_groups():
    while True:
        g = [random.randint(0, 0xFFFF) for _ in range(8)]
        # bias toward zero-runs so :: compression is exercised
        if random.random() < 0.7:
            start = random.randint(0, 6)
            length = random.randint(1, 8 - start)
            for i in range(start, start + length):
                g[i] = 0
        if not is_v4mapped(g):
            return g


def fmt_v6_input(groups):
    """Render groups into a (possibly non-canonical) valid input string."""
    style = random.random()
    parts = []
    for g in groups:
        h = format(g, "x")
        if style < 0.25 and g != 0:  # random leading zeros
            h = h.zfill(random.choice([2, 3, 4]))
        if style > 0.75:  # uppercase
            h = h.upper()
        parts.append(h)
    s = ":".join(parts)
    # randomly collapse one zero-run into :: to vary input form
    if random.random() < 0.5:
        runs = []
        i = 0
        while i < 8:
            if groups[i] == 0:
                j = i
                while j < 8 and groups[j] == 0:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        runs = [r for r in runs if r[1] - r[0] >= 1]
        if runs:
            a, b = random.choice(runs)
            left = parts[:a]
            right = parts[b:]
            s = ":".join(left) + "::" + ":".join(right)
            if not left:
                s = "::" + ":".join(right)
            if not right:
                s = ":".join(left) + "::"
            if not left and not right:
                s = "::"
    return s


def rand_v4_input():
    octs = [random.randint(0, 255) for _ in range(4)]
    return ".".join(str(o) for o in octs)


# ---- valid IPv6 ----
for _ in range(900):
    R.append(fmt_v6_input(rand_groups()))
# ---- valid IPv4 ----
for _ in range(500):
    R.append(rand_v4_input())
# ---- valid CIDR ----
for _ in range(450):
    if random.random() < 0.5:
        R.append(fmt_v6_input(rand_groups()) + "/" + str(random.randint(0, 128)))
    else:
        R.append(rand_v4_input() + "/" + str(random.randint(0, 32)))
# ---- adversarial INVALID ----
bad = [
    "", " ", "::", "g::1", "1::2::3", "12345::1", "1:2:3:4:5:6:7:8:9",
    "1:2:3:4:5:6:7", ":1:2:3:4:5:6:7", "1:2:3:4:5:6:7:", "::1::",
    "192.168.000.1", "256.1.1.1", "1.2.3", "1.2.3.4.5", "01.2.3.4",
    "1.2.3.04", "1.2.3.4 ", " 1.2.3.4", "0xff.0.0.1", "1.2.-3.4",
    "1.2.3.4/33", "1.2.3.4/-1", "1.2.3.4/", "1.2.3.4/0xff", "2001:db8::/129",
    "a::b::c/64", "192.168.001.0/24", "1.2.3.4//8", "fe80::1%eth0",
    "::ffff:1.2.3.4", "1.2.3.256", "abcd", "12:34", "/24", "1.2.3.4/ 8",
]
# random structural corruption
for _ in range(400):
    g = rand_groups()
    s = fmt_v6_input(g)
    op = random.random()
    if op < 0.3:
        s = s.replace(":", "", 1)
    elif op < 0.5:
        s = s + ":" + format(random.randint(0, 0xFFFF), "x")
    elif op < 0.7:
        s = s.replace(s.split(":")[0] or "0", "1" + format(random.randint(0, 0xFFFF), "x"), 1)
    else:
        s = "z" + s
    bad.append(s)
for _ in range(200):
    s = rand_v4_input()
    op = random.random()
    if op < 0.4:
        s = s + "." + str(random.randint(0, 255))
    elif op < 0.7:
        parts = s.split(".")
        parts[random.randint(0, 3)] = str(random.randint(256, 999))
        s = ".".join(parts)
    else:
        s = "0" + s
    bad.append(s)
R.extend(bad)

# Scope: pure IPv4 (dotted) or pure-hex IPv6 only. Exclude IPv4-mapped /
# IPv4-embedded IPv6 (address part containing both '.' and ':') and scoped/zone
# IDs ('%'), which Python renders specially and which are out of scope here.
def _addr_part(s):
    return s.split("/", 1)[0]


def _in_scope(s):
    a = _addr_part(s)
    if "%" in s:
        return False
    if "." in a and ":" in a:
        return False
    return True


R = [s for s in R if _in_scope(s)]

# de-dup preserving order, then shuffle deterministically
seen = set()
uniq = []
for s in R:
    if s not in seen:
        seen.add(s)
        uniq.append(s)
random.shuffle(uniq)

with open("golden.txt", "w", encoding="utf-8", newline="\n") as f:
    for s in uniq:
        # inputs never contain tab or newline by construction
        f.write(f"{s}\t{canon(s)}\n")

n_valid = sum(1 for s in uniq if canon(s) != "INVALID")
print(f"wrote {len(uniq)} cases, {n_valid} valid, {len(uniq)-n_valid} invalid")
