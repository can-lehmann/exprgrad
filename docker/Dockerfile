ARG IMAGE_TAG="1.6.0-alpine"
FROM nimlang/nim:${IMAGE_TAG}

RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/testing >> /etc/apk/repositories
RUN apk add --no-cache \
        llvm-dev \
        openssl \
        pcre \
        ;
ENTRYPOINT ["nimble"]
CMD ["-h"]
