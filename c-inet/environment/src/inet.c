/* inet -- canonicalize an IPv4 / IPv6 address or CIDR network.
 *
 * Read one query per line from stdin; for each, print on stdout either the
 * canonical form or the literal token INVALID, one output line per input line.
 * The full I/O and canonicalization contract is in /app/docs/CONTRACT.md.
 *
 * The verifier compiles this file with:
 *     gcc -O2 -std=c11 -o /app/bin/inet /app/src/inet.c
 * and feeds queries on stdin. Implement canonicalize() below.
 */
#include <stdio.h>
#include <string.h>

/* Write the canonical form of `in` into `out` (a buffer of at least 80 bytes),
 * or the literal "INVALID" if `in` is not a valid, in-scope query.
 * TODO: implement per /app/docs/CONTRACT.md. */
static void canonicalize(const char *in, char *out) {
    (void)in;
    strcpy(out, "INVALID");
}

int main(void) {
    char line[300];
    while (fgets(line, sizeof(line), stdin)) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r'))
            line[--L] = '\0';
        char out[80];
        canonicalize(line, out);
        printf("%s\n", out);
    }
    return 0;
}
