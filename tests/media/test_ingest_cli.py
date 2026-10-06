# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production CPU CLI contracts; synthetic offline input and legacy references."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
        self.assertFalse(result["hardware_decode"])
        self.assertFalse(result["hdr_to_sdr"])
        version,_=self.invoke("version")
        self.assertTrue(version["ffmpeg_license"])
        self.assertEqual(version["version"],"0.1.0")
        self.assertIn(b"media-ingest extract",subprocess.check_output([str(CLI),"--help"]))
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
               ("cfr-asymmetric",["--window"],{"sharpness":True,"window":True})]
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
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cli",type=Path,required=True);parser.add_argument("--runner",type=Path,required=True)
    args,remaining=parser.parse_known_args();CLI=args.cli.resolve();RUNNER=args.runner.resolve()
    unittest.main(argv=[__file__,*remaining])
