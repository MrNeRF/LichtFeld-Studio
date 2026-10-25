# SPDX-License-Identifier: GPL-3.0-or-later
{
  description = "LichtFeld Studio development environment for NixOS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # NVIDIA CUDA Toolkit
      };
      # Nixpkgs' merged toolkit omits nvJPEG, which nvImageCodec links statically.
      cuda = pkgs.symlinkJoin {
        name = "lichtfeld-cuda-12.8";
        paths = [
          pkgs.cudaPackages_12_8.cudatoolkit
        ]
        ++ map (output: pkgs.cudaPackages_12_8.libnvjpeg.${output}) [
          "dev"
          "include"
          "lib"
          "static"
        ];
      };
      environment = pkgs.buildFHSEnv {
        name = "lichtfeld-dev";
        targetPkgs =
          p: with p; [
            gcc14
            gnumake
            cmake
            ninja
            pkg-config
            git
            git-lfs
            curl
            cacert
            zip
            unzip
            which
            file
            patch
            python3
            perl
            ccache
            nasm
            autoconf
            autoconf-archive
            automake
            libtool
            m4
            bison
            flex
            gettext
            gperf
            cuda
            gtk3
            glib
            pango
            harfbuzz
            cairo
            gdk-pixbuf
            atk
            at-spi2-atk
            at-spi2-core
            fontconfig
            freetype
            fribidi
            libepoxy
            pixman
            libthai
            libdatrie
            libffi
            pcre2
            zlib
            bzip2
            brotli
            libpng
            libjpeg
            libtiff
            expat
            graphite2
            libsysprof-capture
            libxft
            libdeflate
            lerc
            zstd
            libwebp
            util-linux
            libselinux
            libsepol
            dbus
            systemd
            libGL
            libGLU
            vulkan-loader
            wayland
            wayland-scanner
            wayland-protocols
            libxkbcommon
            libdecor
            ibus
            libx11
            libxext
            libxcursor
            libxi
            libxinerama
            libxrandr
            libxrender
            libxfixes
            libxdamage
            libxcomposite
            libxtst
            libxau
            libxdmcp
            libxscrnsaver
            libxxf86vm
            libxcb
            xorgproto
            xauth
            xorg-server
          ];
        multiPkgs = null;
        extraOutputsToInstall = [ "dev" ];
        profile = ''
          export PKG_CONFIG_PATH="/usr/lib/pkgconfig:/usr/share/pkgconfig"
          export CC=gcc
          export CXX=g++
          export CUDAHOSTCXX=g++
          export CUDA_PATH=${cuda}
          export CUDAToolkit_ROOT=${cuda}
          export CUDACXX=${cuda}/bin/nvcc
          export VCPKG_ROOT="''${VCPKG_ROOT:-$PWD/vcpkg}"
          export VCPKG_FORCE_SYSTEM_BINARIES=1
          export VCPKG_DISABLE_METRICS=1
          export LFS_VCPKG_REV=817892b0ecbc6af994dd6df999c73faf795adbf2
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:/usr/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
        runScript = "bash";
      };
    in
    {
      devShells.${system}.default = environment.env;
      packages.${system}.default = environment;
      formatter.${system} = pkgs.nixfmt;
    };
}
