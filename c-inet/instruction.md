Implement an IP address canonicalizer in C at `/app/src/inet.c`.

The program reads one query per line from standard input and prints one line of
output per input line: either the query's canonical form or the literal token
`INVALID`. A query is either a bare IPv4/IPv6 address or a CIDR block
`address/prefix`. IPv6 output must follow the RFC 5952 canonical text rules
(lowercase, no leading zeros, longest-run `::` compression), and CIDR blocks must
be reduced to their network address with host bits zeroed.

The full, normative contract — query kinds, the strict IPv4 and IPv6 parsing
rules, the RFC 5952 formatting rules, the CIDR semantics, and exactly which
inputs are out of scope or invalid — is in `/app/docs/CONTRACT.md`. Follow it
precisely; many inputs differ only in subtle edge cases.

The verifier compiles your file with `gcc -O2 -std=c11 -o /app/bin/inet
/app/src/inet.c` and feeds queries on stdin, so your program must build cleanly
with that command and read from standard input. Edit only `/app/src/inet.c`.
