#!/usr/bin/env bash
set -euxo pipefail

if [ ! -d 'ffmpeg-6.1.1-full_build-shared' ]; then
  wget 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-6.1.1-full_build-shared.7z' -O 'ffmpeg.7z'
  7zr x 'ffmpeg.7z'
  rm 'ffmpeg.7z'
fi

wine python -m pip install -r src/requirements.txt pyinstaller==6.4.0
rm -rf build/
wine python -m PyInstaller src/win.spec --noconfirm --distpath dist/release/
wine python -m PyInstaller src/win.spec --noconfirm --distpath dist/debug/ -- --debug

rm -rf out/Search/
cp -r dist/release/Search out/
cp 'dist/debug/Search/Search.exe' 'out/Search/Search debug.exe'
cp 'dist/debug/Search/Index Creator.exe' 'out/Search/Index Creator debug.exe'
cp -r 'ffmpeg-6.1.1-full_build-shared' 'out/Search/ffmpeg'
