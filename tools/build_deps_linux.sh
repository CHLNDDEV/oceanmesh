#!/usr/bin/env bash
set -euxo pipefail

PREFIX="${1:-/opt/om}"

GMP_VERSION="6.3.0"
MPFR_VERSION="4.2.1"

GMP_URL="https://ftp.gnu.org/gnu/gmp/gmp-${GMP_VERSION}.tar.xz"
MPFR_URL="https://ftp.gnu.org/gnu/mpfr/mpfr-${MPFR_VERSION}.tar.xz"

install_build_prereqs() {
  if command -v yum >/dev/null 2>&1; then
    yum -y install m4
  elif command -v dnf >/dev/null 2>&1; then
    dnf -y install m4
  else
    echo "Neither yum nor dnf found; cannot install build prerequisites" >&2
    exit 2
  fi
}

build_one() {
  local name="$1"
  local version="$2"
  local url="$3"

  local workdir
  workdir="/tmp/build-${name}"
  rm -rf "$workdir"
  mkdir -p "$workdir"
  pushd "$workdir" >/dev/null

  curl -fsSL "$url" -o "${name}-${version}.tar.xz"
  tar -xf "${name}-${version}.tar.xz"

  pushd "${name}-${version}" >/dev/null

  if [ "$name" = "gmp" ]; then
    ./configure \
      --prefix="$PREFIX" \
      --enable-shared \
      --disable-static \
      --enable-cxx \
      --with-pic
  elif [ "$name" = "mpfr" ]; then
    ./configure \
      --prefix="$PREFIX" \
      --with-gmp="$PREFIX" \
      --enable-shared \
      --disable-static \
      --with-pic
  else
    echo "Unknown dep: $name" >&2
    exit 2
  fi

  make -j"$(nproc)"
  make install

  popd >/dev/null
  popd >/dev/null
}

mkdir -p "$PREFIX"
install_build_prereqs

build_one gmp "$GMP_VERSION" "$GMP_URL"
build_one mpfr "$MPFR_VERSION" "$MPFR_URL"

echo "Built deps installed into: $PREFIX"
ls -la "$PREFIX/lib" | head
