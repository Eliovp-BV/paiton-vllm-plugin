#!/bin/sh
set -eu
export PATH=/opt/meeting-media/bin:$PATH
export PKG_CONFIG_PATH=/opt/meeting-media/lib/pkgconfig
export LD_LIBRARY_PATH=/opt/meeting-media/lib:${LD_LIBRARY_PATH:-}
export CC=gcc
export CXX=g++
export CFLAGS='-O2 -fPIC'
export LDFLAGS='-Wl,-rpath,/opt/meeting-media/lib'
mkdir -p /build /wheels
cd /build
for archive in nasm-2.16.03.tar.xz lame_3.100.orig.tar.gz opus-1.6.1.tar.gz ffmpeg-8.1.2.tar.xz av-18.1.0.tar.gz; do tar xf /sources/$archive; done
cd /build/nasm-2.16.03
./configure --prefix=/opt/meeting-media
make -j2
make install
cd /build/lame-3.100
./configure --prefix=/opt/meeting-media --enable-shared --disable-static --disable-frontend
make -j2
make install
cd /build/opus-1.6.1
./configure --prefix=/opt/meeting-media --enable-shared --disable-static --disable-doc --disable-extra-programs
make -j2
make install
cd /build/ffmpeg-8.1.2
./configure --prefix=/opt/meeting-media --enable-shared --disable-static --enable-version3 --disable-gpl --disable-nonfree --disable-autodetect --disable-network --disable-programs --disable-doc --disable-debug --enable-libmp3lame --enable-libopus --extra-cflags=-I/opt/meeting-media/include --extra-ldflags="-L/opt/meeting-media/lib -Wl,-rpath,/opt/meeting-media/lib"
make -j2
make install
