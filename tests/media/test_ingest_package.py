# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Install, relocate and consume the CPU package without repository include paths."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("media_fixtures",ROOT/"scripts/prepare_media_fixtures.py")
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
def run(args):
    result=subprocess.run(list(map(str,args)),capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout.decode("utf-8",errors="replace")+result.stderr.decode("utf-8",errors="replace"))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--cmake",default="cmake")
    parser.add_argument("--generator",default="Ninja")
    parser.add_argument("--compiler",default="")
    parser.add_argument("--resource-compiler",default="")
    parser.add_argument("--manifest-tool",default="")
    parser.add_argument("--toolchain",default="")
    parser.add_argument("--triplet",default="")
    parser.add_argument("--packages",default="")
    parser.add_argument("--prefix",default="")
    parser.add_argument("--deployment-target",default="")
    parser.add_argument("--osx-architectures",default="")
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lfs-media-package-") as temp:
        work=Path(temp);staged=work/"installed";relocated=work/"relocated";build=work/"consumer-build"
        run([args.cmake,"--install",args.build,"--prefix",staged,"--config","Release"])
        assert not (staged/"include/io/video_frame_extractor.hpp").exists()
        assert not (staged/"include/core/tensor.hpp").exists()
        assert (staged/"share/lichtfeld-media/LICENSE").is_file()
        assert (staged/"share/lichtfeld-media/licenses/stb-image-license.txt").is_file()
        # Both paths are freshly-created children of the test's own temporary root.
        assert staged.resolve().parent==work.resolve() and relocated.resolve().parent==work.resolve()
        shutil.move(staged,relocated)
        command=[args.cmake,"-S",ROOT/"tests/media/consumer","-B",build,"-G",args.generator,"-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_PREFIX_PATH="+str(relocated)+((';'+args.prefix) if args.prefix else ''),"-DVCPKG_MANIFEST_INSTALL=OFF"]
        if args.compiler: command.append("-DCMAKE_CXX_COMPILER="+args.compiler)
        if args.resource_compiler: command.append("-DCMAKE_RC_COMPILER="+args.resource_compiler)
        if args.manifest_tool: command.append("-DCMAKE_MT="+args.manifest_tool)
        if args.toolchain: command.append("-DCMAKE_TOOLCHAIN_FILE="+args.toolchain)
        if args.triplet: command.append("-DVCPKG_TARGET_TRIPLET="+args.triplet)
        if args.packages: command.append("-DVCPKG_INSTALLED_DIR="+args.packages)
        if args.deployment_target: command.append("-DCMAKE_OSX_DEPLOYMENT_TARGET="+args.deployment_target)
        if args.osx_architectures: command.append("-DCMAKE_OSX_ARCHITECTURES="+args.osx_architectures)
        run(command)
        run([args.cmake,"--build",build,"--config","Release","--parallel","2"])
        corpus=work/"corpus é 日本語"
        fixtures.prepare(corpus,os.environ.get("LFS_MEDIA_TEST_FFMPEG","ffmpeg"),os.environ.get("LFS_MEDIA_TEST_FFPROBE","ffprobe"))
        executable=build/("media_consumer.exe" if os.name=="nt" else "media_consumer")
        if not executable.exists(): executable=build/"Release"/executable.name
        run([executable,corpus/"cfr-asymmetric.nut",work/"external-output"])
        print("Relocated package: configure, build, public probe/extraction, errors, cancellation and ownership passed")
if __name__=="__main__": main()
