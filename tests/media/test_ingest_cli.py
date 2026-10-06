# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production CPU CLI contracts; synthetic offline input and legacy references."""
import argparse
import ctypes
import importlib.util
import json
import os
import platform
import re
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("media_fixtures",ROOT/"scripts/prepare_media_fixtures.py")
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
FFMPEG=os.environ.get("LFS_MEDIA_TEST_FFMPEG","ffmpeg")
FFPROBE=os.environ.get("LFS_MEDIA_TEST_FFPROBE","ffprobe")
CLI=None
RUNNER=None
VERSION_HEADER=None
CONSUMERS=[]
SYMBOLS_TOOL=None
BUILD_DIR=None
BUILD_CONFIG=None

def defined_symbols(path, exports=False):
    if sys.platform == "win32":
        tool=SYMBOLS_TOOL or shutil.which("dumpbin")
        if not tool:
            raise RuntimeError("MSVC symbol inspection tool is required")
        command=[str(tool)]
        if Path(tool).name.lower() in ("link.exe", "link"):
            command.append("/dump")
        command.extend(["/exports", str(path)])
        output=subprocess.check_output(command,text=True,errors="replace")
        return set(re.findall(r"^\s+\d+\s+[0-9A-Fa-f]+\s+(?:[0-9A-Fa-f]+\s+)?([A-Za-z_][A-Za-z0-9_]*)",output,re.M))
    tool=SYMBOLS_TOOL or shutil.which("nm")
    if not tool:
        raise RuntimeError("nm is required for provider ownership contracts")
    flags=["-gU"] if sys.platform == "darwin" else (["-D","--defined-only"] if exports else ["--defined-only"])
    output=subprocess.check_output([str(tool),*flags,str(path)],text=True,errors="replace")
    names=set()
    for line in output.splitlines():
        fields=line.split()
        if len(fields)>=3:
            name=fields[-1].split("@")[0]
            if sys.platform == "darwin" and name.startswith("_"):
                name=name[1:]
            # Local static inline helpers in public FFmpeg headers are not
            # independent archive implementations in a consumer.
            if exports or fields[-2].isupper() or name.startswith("ff_"):
                names.add(name)
    return names

def ffmpeg_public(name):
    return re.match(r"^(av|avcodec|avformat|avutil|avfilter|avdevice|avio|swscale|swresample|sws|swr)_",name)
class IngestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix="lfs-cli-contratti-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name)
        cls.corpus=cls.root/"media é 日本語"
        fixtures.prepare(cls.corpus,FFMPEG,FFPROBE)
    def invoke(self,*args,code=0):
        result=subprocess.run([str(CLI),*map(str,args)],capture_output=True,timeout=30)
        self.assertEqual(result.returncode,code,result.stderr.decode("utf-8",errors="replace")+result.stdout.decode("utf-8",errors="replace"))
        return json.loads(result.stdout),result.stderr
    def source(self,name="cfr-asymmetric"):
        return self.corpus/(name+".nut")
    def test_capabilities_and_help(self):
        result,_=self.invoke("capabilities")
        self.assertTrue(result["software_decode"])
        self.assertEqual(result["hardware_decode"],sys.platform == "darwin")
        self.assertFalse(result["hdr_to_sdr"])
        version,_=self.invoke("version")
        self.assertTrue(version["ffmpeg_license"])
        self.assertTrue(version["version"])
        expected=re.search(r'^#define GIT_TAGGED_VERSION "(.*)"',VERSION_HEADER.read_text(encoding="utf-8"),re.MULTILINE)
        self.assertIsNotNone(expected)
        self.assertEqual(version["version"],expected.group(1))
        self.assertIn(b"--hdr-to-sdr",subprocess.check_output([str(CLI),"--help"]))
    @unittest.skipUnless(sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64","amd64"),
                         "ELF x86 FFmpeg assembly symbol isolation")
    def test_elf_ffmpeg_assembly_data_stays_private(self):
        library=ctypes.CDLL(str(CLI.parent/"liblfs_media.so"))
        # These direct-reference assembly constants must not be interposable.
        # Public MediaIngest APIs remain exercised by the CLI contracts below.
        for symbol in ("ff_pw_9","ff_pw_512"):
            with self.subTest(symbol=symbol):
                with self.assertRaises(ValueError):
                    ctypes.c_uint64.in_dll(library,symbol)
    def test_ffmpeg_has_one_explicit_provider(self):
        # Release binaries may be stripped. Check the generated final link
        # commands as well as binary symbols; ownership cannot depend on nm
        # retaining a consumer's ordinary symbol table.
        reply=BUILD_DIR/".cmake/api/v1/reply"
        indexes=list(reply.glob("index-*.json"))
        self.assertTrue(indexes,"Root CMake codemodel is required")
        index=json.loads(max(indexes,key=lambda p:p.stat().st_mtime_ns).read_text(encoding="utf-8"))
        model_ref=next(item for item in index["objects"] if item["kind"]=="codemodel")
        model=json.loads((reply/model_ref["jsonFile"]).read_text(encoding="utf-8"))
        configuration=next(item for item in model["configurations"] if item["name"]==BUILD_CONFIG)
        providers=set()
        archives=re.compile(r"(?:lib)?(?:avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample)d?\.(?:a|lib)\b|(?<![\w-])-l(?:avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample)\b",re.I)
        for ref in configuration["targets"]:
            target=json.loads((reply/ref["jsonFile"]).read_text(encoding="utf-8"))
            fragments=target.get("link",{}).get("commandFragments",[])
            ffmpeg=[item["fragment"] for item in fragments if archives.search(item["fragment"])]
            if ffmpeg:
                providers.add(target["name"])
                self.assertEqual(target["name"],"lfs_media",f"FFmpeg linked independently by {target['name']}: {ffmpeg}")
        self.assertEqual(providers,{"lfs_media"})
        filename="lfs_media.dll" if sys.platform=="win32" else ("liblfs_media.dylib" if sys.platform=="darwin" else "liblfs_media.so")
        provider=CLI.parent/filename
        exports=defined_symbols(provider,exports=True)
        required={"avformat_open_input","avcodec_alloc_context3","avcodec_send_frame",
                  "av_frame_alloc","av_packet_alloc","sws_scale","swr_alloc","swscale_version","swresample_version"}
        self.assertTrue(required<=exports,f"Missing public provider APIs: {required-exports}")
        self.assertFalse({name for name in exports if name.startswith(("ff_","avpriv_"))},
                         "Internal FFmpeg symbols must not escape the provider")
        self.assertTrue(CONSUMERS,"Root application consumers must be supplied")
        for consumer in [CLI,RUNNER,*CONSUMERS]:
            with self.subTest(consumer=consumer.name):
                self.assertTrue(consumer.is_file(),f"Build the root media_contracts target: missing {consumer}")
                symbols=defined_symbols(consumer)
                copies={name for name in symbols if ffmpeg_public(name) or name.startswith(("ff_","avpriv_"))}
                self.assertFalse(copies,f"FFmpeg implementation leaked into {consumer}: {sorted(copies)[:10]}")
                if sys.platform=="win32":
                    tool=SYMBOLS_TOOL or shutil.which("dumpbin")
                    command=[str(tool)]
                    if Path(tool).name.lower() in ("link.exe","link"):
                        command.append("/dump")
                    imports=subprocess.check_output([*command,"/imports",str(consumer)],text=True,errors="replace")
                    self.assertFalse(re.findall(r"\b(?:avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample)-\d+\.dll\b",imports,re.I),
                                     "Consumers must import FFmpeg through lfs_media")
    def test_probe_rational_inventory_and_unicode_no_output(self):
        before={p.name:p.read_bytes() for p in self.corpus.iterdir()}
        result,_=self.invoke("probe",self.source())
        media=result["media"]
        ref=json.loads(subprocess.check_output([FFPROBE,"-v","error","-show_streams","-of","json",str(self.source())]))["streams"][0]
        self.assertEqual(media["streams"][0]["time_base"],[int(x) for x in ref["time_base"].split("/")])
        self.assertEqual(media["streams"][0]["width"],64)
        headers,_=self.invoke("probe",self.source(),"--headers-only")
        self.assertFalse(headers["media"]["stream_info_probed"])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.corpus.iterdir()})
    def test_extract_matches_legacy_pixels_names_metadata(self):
        cases=[("cfr-asymmetric",[],{}),("vfr-asymmetric",[],{}),
               ("cfr-asymmetric",["--rotate","90"],{"rotation":90}),
               ("cfr-asymmetric",["--scale","0.5"],{"scale":0.5}),
               ("cfr-asymmetric",["--size","40","24"],{"width":40,"height":24}),
               ("cfr-asymmetric",["--start","0.1"],{"start":0.1}),
               ("cfr-asymmetric",["--format","jpeg"],{"format":"jpg"}),
               ("cfr-asymmetric",["--window"],{"sharpness":True,"window":True}),
               ("cfr-asymmetric",["--window","--algorithm","laplacian"],{"sharpness":True,"window":True,"algorithm":"laplacian"}),
               ("cfr-asymmetric",["--window","--algorithm","tenengrad"],{"sharpness":True,"window":True,"algorithm":"tenengrad"})]
        for name,options,legacy_options in cases:
            with self.subTest(options=options):
                work=Path(tempfile.mkdtemp(dir=self.root));output=work/"CLI é 日本語";legacy=work/"legacy"
                actual,stderr=self.invoke("extract",self.source(name),"--output",output,"--interval","1","--end","0.35","--name","frame_%03d","--metadata",*options)
                request={"input":str(self.source(name)),"output":str(legacy),"interval":1,"end":0.35,**legacy_options}
                path=work/"request.json";path.write_text(json.dumps(request,ensure_ascii=False),encoding="utf-8")
                ref=subprocess.run([str(RUNNER),str(path)],capture_output=True,timeout=30)
                self.assertEqual(ref.returncode,0,ref.stderr)
                self.assertTrue(json.loads(ref.stdout)["success"])
                self.assertEqual({p.name:p.read_bytes() for p in output.iterdir() if p.suffix in (".png",".jpg")},
                                 {p.name:p.read_bytes() for p in legacy.iterdir() if p.suffix in (".png",".jpg")})
                metadata=json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
                reference=json.loads((legacy/"extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["frames"],reference["frames"])
                self.assertEqual(actual["frames_accepted"],len(metadata["frames"]))
                progress=[json.loads(line) for line in stderr.splitlines() if line.startswith(b'{')]
                self.assertTrue(progress)
    def test_fps_vfr_quiet_and_metadata_off(self):
        output=self.root/"fps-output"
        actual,stderr=self.invoke("extract",self.source("vfr-asymmetric"),"--output",output,"--fps","5","--end","1.5","--quiet")
        self.assertTrue(actual["success"])
        self.assertFalse((output/"extraction_metadata.json").exists())
        self.assertNotIn(b'"event":"progress"',stderr)
        self.assertEqual(actual["frames_accepted"],len(list(output.glob("*.png"))))
    def test_invalid_arguments_fail_before_output(self):
        for flags in (["--fps","0"],["--fps","nan"],["--interval","0"],["--scale","0"],
                      ["--size","0","32"],["--rotate","45"],["--quality","101"],
                      ["--fps","1","--interval","1"],["--scale","1","--size","64","32"],
                      ["--format","other"],["--unknown"],["--fps"]):
            with self.subTest(flags=flags):
                output=self.root/"invalid-output"
                result,_=self.invoke("extract",self.source(),"--output",output,*flags,code=2)
                self.assertEqual(result["error"]["code"],"InvalidArgument")
                self.assertFalse(output.exists())
    def test_missing_input_and_hdr_capability(self):
        result,_=self.invoke("probe",self.root/"missing.nut",code=3)
        self.assertEqual(result["error"]["code"],"NotFound")
        output=self.root/"unavailable-hdr"
        result,_=self.invoke("extract",self.source(),"--output",output,"--hdr-to-sdr",code=3)
        self.assertEqual(result["error"]["code"],"Unsupported")
        self.assertFalse(output.exists())
    def test_writer_failure_retains_count_and_files(self):
        output=self.root/"blocked-file";output.mkdir();(output/"frame_2.png").mkdir()
        result,_=self.invoke("extract",self.source(),"--output",output,"--interval","1","--end","0.35",code=3)
        self.assertTrue((output/"frame_1.png").is_file())
        fields=[c["fields"] for c in result["error"]["context"]]
        self.assertEqual(next(f["frames_accepted"] for f in fields if "frames_accepted" in f),1)
    def test_filesystem_error_keeps_utf8_native_status(self):
        output=self.root/"blocked é 日本語";output.write_bytes(b"keep")
        result,_=self.invoke("extract",self.source(),"--output",output,code=3)
        self.assertIn(str(output),result["error"]["message"])
        self.assertIn("native",result["error"])
        self.assertEqual(output.read_bytes(),b"keep")
    @unittest.skipIf(sys.platform == "win32", "Windows terminate() does not deliver a CRT SIGTERM")
    def test_sigterm_cancels_and_preserves_written_frames(self):
        source=self.root/"long.nut"
        subprocess.run([FFMPEG,"-v","error","-f","lavfi","-i","color=size=64x48:rate=100:duration=120",
                        "-c:v","ffv1",str(source)],check=True,timeout=30)
        output=self.root/"cancelled"
        with tempfile.TemporaryFile() as stderr:
            process=subprocess.Popen([str(CLI),"extract",str(source),"--output",str(output),"--interval","1"],
                                     stdout=subprocess.PIPE,stderr=stderr)
            try:
                import time
                deadline=time.monotonic()+20
                while not (output/"frame_1.png").exists() and process.poll() is None and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertTrue((output/"frame_1.png").exists(),"extraction must start before cancellation")
                process.send_signal(signal.SIGTERM)
                stdout,_=process.communicate(timeout=20)
                self.assertEqual(process.returncode,130,stdout)
                result=json.loads(stdout)
                self.assertEqual(result["error"]["code"],"Cancelled")
                fields=[c["fields"] for c in result["error"]["context"]]
                accepted=next(f["frames_accepted"] for f in fields if "frames_accepted" in f)
                self.assertEqual(accepted,len(list(output.glob("*.png"))))
                self.assertGreater(accepted,0)
            finally:
                if process.poll() is None:
                    process.kill();process.communicate()
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cli",type=Path,required=True);parser.add_argument("--runner",type=Path,required=True)
    parser.add_argument("--version-header",type=Path,required=True)
    parser.add_argument("--application",type=Path,required=True)
    parser.add_argument("--visualizer",type=Path,required=True)
    parser.add_argument("--python-module",type=Path,required=True)
    parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--symbols-tool",type=Path)
    args,remaining=parser.parse_known_args();CLI=args.cli.resolve();RUNNER=args.runner.resolve();VERSION_HEADER=args.version_header.resolve()
    CONSUMERS=[args.application.resolve(),args.visualizer.resolve(),args.python_module.resolve()]
    SYMBOLS_TOOL=args.symbols_tool
    BUILD_DIR=args.build_dir.resolve();BUILD_CONFIG=args.config
    unittest.main(argv=[__file__,*remaining])
