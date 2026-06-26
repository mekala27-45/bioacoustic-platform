# inet — address canonicalization contract

`inet` reads one **query** per line from standard input and writes one line of
output per input line: either the query's **canonical form** or the literal
token `INVALID`. Trailing `\r` and `\n` are stripped from each input line before
processing; nothing else is trimmed (a leading or trailing space makes a query
`INVALID`). Output lines are terminated by a single `\n`.

The verifier compiles your program with

```
gcc -O2 -std=c11 -o /app/bin/inet /app/src/inet.c
```

and pipes queries on stdin. Implement the function in `/app/src/inet.c`.

## Query kinds

A query is one of:

- a **bare address** — an IPv4 or IPv6 address, or
- a **CIDR block** — `address/prefix`, where the part before the first `/` is an
  address and the part after it is the prefix length.

A query that contains more than one `/`, or a `/` with an empty address or empty
prefix, is `INVALID`.

## Scope

Only two address shapes are in scope:

- **IPv4**: dotted-decimal, four octets, e.g. `192.168.0.1`.
- **IPv6**: pure hexadecimal groups, e.g. `2001:db8::1`.

IPv4-mapped / IPv4-embedded IPv6 (an address whose textual form contains both a
`.` and a `:`, e.g. `::ffff:1.2.3.4`) and scoped/zone-id addresses (containing
`%`) are **out of scope and always `INVALID`**.

## IPv4 parsing

An IPv4 address is exactly four octets separated by `.`. Each octet:

- is 1 to 3 decimal digits,
- has **no leading zero** (so `0` is valid but `00` and `01` are not),
- has value 0–255.

Anything else (wrong octet count, empty octet, non-digit, value > 255, leading or
trailing characters) is `INVALID`.

## IPv6 parsing

An IPv6 address is up to eight 16-bit groups written in hexadecimal and separated
by `:`. Parsing rules:

- Each group is 1 to 4 hex digits (`0`–`9`, `a`–`f`, `A`–`F`). Leading zeros are
  allowed on input (`00ab` parses as `0x00ab`). A group with 5+ hex digits is
  `INVALID`.
- The token `::` may appear **at most once** and stands for one or more groups of
  zeros, chosen so the address has exactly eight groups total. `::` may represent
  a single zero group on input.
- An address without `::` must have exactly eight groups.
- A single leading `:` not part of `::` (e.g. `:1:2:...`), a single trailing `:`
  (e.g. `1:2:...:`), an empty group, two `::`, or more than eight groups present
  is `INVALID`.

## RFC 5952 canonical form (IPv6 output)

Canonicalize a parsed IPv6 address to text as follows:

- Each group is printed in **lowercase hex with no leading zeros** (a zero group
  is `0`).
- The **longest run of two or more consecutive zero groups** is replaced by `::`.
  A single zero group is **never** shortened to `::`. If several runs share the
  maximum length, compress the **leftmost** one. There is at most one `::`.
- Examples: `0:0:0:0:0:0:0:0` → `::`; `0:0:0:0:0:0:0:1` → `::1`;
  `1:0:0:0:0:0:0:0` → `1::`; `1:0:0:0:1:0:0:1` → `1::1:0:0:1`;
  `1:0:0:1:0:0:0:1` → `1:0:0:1::1`;
  `1:2:3:4:5:0:7:8` → `1:2:3:4:5:0:7:8` (single zero group, not compressed).

IPv4 output is the four octets in decimal, dot-separated, with no leading zeros.

## CIDR blocks

For `address/prefix`:

- Parse `address` exactly as above (IPv4 or IPv6).
- `prefix` is a decimal integer with **no leading zero** (except the literal `0`),
  in range 0–32 for IPv4 and 0–128 for IPv6. Anything else is `INVALID`.
- Zero all **host bits** (the bits after the first `prefix` bits) of the address,
  producing the **network address**.
- Output the canonical form of the network address, then `/`, then the prefix in
  decimal. Examples: `192.168.1.5/24` → `192.168.1.0/24`;
  `2001:db8::1/32` → `2001:db8::/32`; `10.0.0.0/8` → `10.0.0.0/8`.

Any query that is not a valid, in-scope bare address or CIDR block produces the
single line `INVALID`.
