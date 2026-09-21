/* SPDX-License-Identifier: Apache-2.0
 * Linux/glibc offline transport guard for the qualified, trusted serving stack.
 * Prebuilt in the execution bundle; never compiled by the product launcher.
 * Unix IPC, loopback and responses on accepted serving sockets remain usable.
 * This is an offline runtime control, not a sandbox for hostile native code.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

static int active(void) {
    const char *value = getenv("PAITON_OFFLINE_ACTIVE");
    return value && strcmp(value, "1") == 0;
}

int paiton_offline_guard_version(void) { return 1; }

static int allowed(const struct sockaddr *address, socklen_t length) {
    if (!active()) return 1;
    if (!address || length < sizeof(sa_family_t)) return 0;
    if (address->sa_family == AF_UNIX) return 1;
    if (address->sa_family == AF_INET && length >= sizeof(struct sockaddr_in)) {
        const struct sockaddr_in *a = (const struct sockaddr_in *)address;
        return (ntohl(a->sin_addr.s_addr) >> 24) == 127;
    }
    if (address->sa_family == AF_INET6 && length >= sizeof(struct sockaddr_in6)) {
        const struct in6_addr *a = &((const struct sockaddr_in6 *)address)->sin6_addr;
        if (IN6_IS_ADDR_LOOPBACK(a)) return 1;
        return IN6_IS_ADDR_V4MAPPED(a) && a->s6_addr[12] == 127;
    }
    return 0;
}

int connect(int fd, const struct sockaddr *address, socklen_t length) {
    if (!allowed(address, length)) { errno = ENETUNREACH; return -1; }
    int (*next)(int, const struct sockaddr *, socklen_t) = dlsym(RTLD_NEXT, "connect");
    if (!next) { errno = ENOSYS; return -1; }
    return next(fd, address, length);
}

ssize_t sendto(int fd, const void *buf, size_t n, int flags,
               const struct sockaddr *address, socklen_t length) {
    if (address && !allowed(address, length)) { errno = ENETUNREACH; return -1; }
    ssize_t (*next)(int, const void *, size_t, int, const struct sockaddr *, socklen_t)
        = dlsym(RTLD_NEXT, "sendto");
    if (!next) { errno = ENOSYS; return -1; }
    return next(fd, buf, n, flags, address, length);
}

ssize_t sendmsg(int fd, const struct msghdr *message, int flags) {
    if (message && message->msg_name && !allowed(message->msg_name, message->msg_namelen)) {
        errno = ENETUNREACH; return -1;
    }
    ssize_t (*next)(int, const struct msghdr *, int) = dlsym(RTLD_NEXT, "sendmsg");
    if (!next) { errno = ENOSYS; return -1; }
    return next(fd, message, flags);
}

int sendmmsg(int fd, struct mmsghdr *messages, unsigned int count, int flags) {
    for (unsigned int i = 0; active() && i < count; ++i) {
        const struct msghdr *m = &messages[i].msg_hdr;
        if (m->msg_name && !allowed(m->msg_name, m->msg_namelen)) {
            errno = ENETUNREACH; return -1;
        }
    }
    int (*next)(int, struct mmsghdr *, unsigned int, int) = dlsym(RTLD_NEXT, "sendmmsg");
    if (!next) { errno = ENOSYS; return -1; }
    return next(fd, messages, count, flags);
}

static int local_name(const char *name) {
    if (!name || !*name || strcmp(name, "localhost") == 0) return 1;
    struct sockaddr_in a4 = {.sin_family = AF_INET};
    struct sockaddr_in6 a6 = {.sin6_family = AF_INET6};
    if (inet_pton(AF_INET, name, &a4.sin_addr) == 1)
        return allowed((const struct sockaddr *)&a4, sizeof(a4));
    if (inet_pton(AF_INET6, name, &a6.sin6_addr) == 1)
        return allowed((const struct sockaddr *)&a6, sizeof(a6));
    return 0;
}

int getaddrinfo(const char *name, const char *service, const struct addrinfo *hints,
                struct addrinfo **result) {
    if (active() && !local_name(name)) return EAI_NONAME;
    int (*next)(const char *, const char *, const struct addrinfo *, struct addrinfo **)
        = dlsym(RTLD_NEXT, "getaddrinfo");
    if (!next) return EAI_SYSTEM;
    return next(name, service, hints, result);
}

struct hostent *gethostbyname(const char *name) {
    if (active() && !local_name(name)) { h_errno = HOST_NOT_FOUND; return NULL; }
    struct hostent *(*next)(const char *) = dlsym(RTLD_NEXT, "gethostbyname");
    return next ? next(name) : NULL;
}
