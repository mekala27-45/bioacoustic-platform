#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/src /app/bin

cat > /app/src/inet.c <<'INET_C_EOF'
/* inet -- strict IPv4/IPv6 parse + RFC 5952 canonicalization + CIDR network. */
#include <stdio.h>
#include <string.h>

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int parse_prefix(const char *s, int maxbits, int *out) {
    if (!*s) return 0;
    if (s[0] == '0' && s[1] != '\0') return 0;
    int v = 0;
    for (const char *p = s; *p; p++) {
        if (*p < '0' || *p > '9') return 0;
        v = v * 10 + (*p - '0');
        if (v > 1000) return 0;
    }
    if (v < 0 || v > maxbits) return 0;
    *out = v;
    return 1;
}
static int parse_ipv4(const char *s, unsigned char b[4]) {
    const char *p = s;
    for (int oct = 0; oct < 4; oct++) {
        if (oct > 0) { if (*p != '.') return 0; p++; }
        const char *start = p;
        int v = 0, n = 0;
        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); n++; p++; if (n > 3) return 0; }
        if (n == 0) return 0;
        if (n > 1 && start[0] == '0') return 0;
        if (v > 255) return 0;
        b[oct] = (unsigned char)v;
    }
    return *p == '\0';
}
static int parse_ipv6(const char *s, unsigned char b[16]) {
    unsigned short groups[8];
    int ngroups = 0, dcolon = -1;
    const char *p = s;
    if (p[0] == ':') {
        if (p[1] != ':') return 0;
        dcolon = 0; p += 2;
        if (*p == '\0') { memset(b, 0, 16); return 1; }
    }
    for (;;) {
        int v = 0, n = 0;
        while (hexval(*p) >= 0) { v = v * 16 + hexval(*p); n++; p++; if (n > 4) return 0; }
        if (n == 0) return 0;
        if (ngroups >= 8) return 0;
        groups[ngroups++] = (unsigned short)v;
        if (*p == '\0') break;
        if (*p != ':') return 0;
        p++;
        if (*p == ':') { if (dcolon >= 0) return 0; dcolon = ngroups; p++; if (*p == '\0') break; }
        else if (*p == '\0') return 0;
    }
    unsigned short full[8];
    if (dcolon < 0) {
        if (ngroups != 8) return 0;
        for (int i = 0; i < 8; i++) full[i] = groups[i];
    } else {
        int zeros = 8 - ngroups;
        if (zeros < 1) return 0;
        int idx = 0;
        for (int i = 0; i < dcolon; i++) full[idx++] = groups[i];
        for (int i = 0; i < zeros; i++) full[idx++] = 0;
        for (int i = dcolon; i < ngroups; i++) full[idx++] = groups[i];
    }
    for (int i = 0; i < 8; i++) { b[2 * i] = full[i] >> 8; b[2 * i + 1] = full[i] & 0xff; }
    return 1;
}
static void mask_bits(unsigned char *b, int nbytes, int prefix) {
    for (int i = 0; i < nbytes; i++) {
        int bitstart = i * 8;
        if (bitstart >= prefix) b[i] = 0;
        else if (bitstart + 8 <= prefix) {}
        else { int keep = prefix - bitstart; b[i] &= (unsigned char)(0xFF << (8 - keep)); }
    }
}
static void fmt_ipv4(const unsigned char b[4], char *out) { sprintf(out, "%d.%d.%d.%d", b[0], b[1], b[2], b[3]); }
static void fmt_ipv6(const unsigned char b[16], char *out) {
    unsigned short g[8];
    for (int i = 0; i < 8; i++) g[i] = (b[2 * i] << 8) | b[2 * i + 1];
    int bs = -1, bl = 0, i = 0;
    while (i < 8) { if (g[i] == 0) { int j = i; while (j < 8 && g[j] == 0) j++; if (j - i > bl) { bl = j - i; bs = i; } i = j; } else i++; }
    if (bl < 2) bs = -1;
    char *o = out;
    if (bs < 0) { for (i = 0; i < 8; i++) { if (i) *o++ = ':'; o += sprintf(o, "%x", g[i]); } }
    else {
        for (i = 0; i < bs; i++) { if (i) *o++ = ':'; o += sprintf(o, "%x", g[i]); }
        *o++ = ':'; *o++ = ':';
        int fa = 1;
        for (i = bs + bl; i < 8; i++) { if (!fa) *o++ = ':'; o += sprintf(o, "%x", g[i]); fa = 0; }
    }
    *o = '\0';
}
static void canonicalize(const char *in, char *out) {
    const char *slash = strchr(in, '/');
    char addr[200];
    int prefix = 0, is_cidr = slash != 0;
    if (is_cidr) {
        size_t a = slash - in;
        if (a >= sizeof(addr)) { strcpy(out, "INVALID"); return; }
        memcpy(addr, in, a); addr[a] = '\0';
    } else {
        if (strlen(in) >= sizeof(addr)) { strcpy(out, "INVALID"); return; }
        strcpy(addr, in);
    }
    int has_dot = strchr(addr, '.') != 0, has_colon = strchr(addr, ':') != 0;
    unsigned char b[16];
    int nbytes, maxbits;
    if (has_dot && !has_colon) { if (!parse_ipv4(addr, b)) { strcpy(out, "INVALID"); return; } nbytes = 4; maxbits = 32; }
    else if (has_colon && !has_dot) { if (!parse_ipv6(addr, b)) { strcpy(out, "INVALID"); return; } nbytes = 16; maxbits = 128; }
    else { strcpy(out, "INVALID"); return; }
    if (is_cidr) {
        if (!parse_prefix(slash + 1, maxbits, &prefix)) { strcpy(out, "INVALID"); return; }
        mask_bits(b, nbytes, prefix);
    }
    char tmp[64];
    if (nbytes == 4) fmt_ipv4(b, tmp); else fmt_ipv6(b, tmp);
    if (is_cidr) sprintf(out, "%s/%d", tmp, prefix); else strcpy(out, tmp);
}
int main(void) {
    char line[300];
    while (fgets(line, sizeof(line), stdin)) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = '\0';
        char out[80];
        canonicalize(line, out);
        printf("%s\n", out);
    }
    return 0;
}
INET_C_EOF

gcc -O2 -std=c11 -Wall -o /app/bin/inet /app/src/inet.c

# Sanity smoke: a few known canonicalizations.
printf '2001:0db8:0000:0000:0000:0000:0000:0001\n192.168.1.5/24\n00:00:00:00:00:00:00:00\n1.2.3.04\n' | /app/bin/inet
