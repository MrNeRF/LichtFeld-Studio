# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline integer SDR references; independent binary EXR and FFmpeg readers."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe")

def run(command, data=None):
    result = subprocess.run(list(map(str, command)), input=data, capture_output=True, timeout=60)
    if result.returncode:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout

def uncompressed_exr(path):
    """Minimal independent reader for the single-part uncompressed test profile."""
    data = path.read_bytes()
    magic, version = struct.unpack_from("<II", data)
    assert magic == 20000630 and version == 2
    offset = 8
    attrs = {}
    def string():
        nonlocal offset
        end = data.index(0, offset)
        text = data[offset:end].decode("ascii")
        offset = end + 1
        return text
    while True:
        name = string()
        if not name:
            break
        kind = string()
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        attrs[name] = (kind, data[offset:offset+size])
        offset += size
    assert attrs["compression"][1] == b"\0"
    xmin, ymin, xmax, ymax = struct.unpack("<4i", attrs["dataWindow"][1])
    width, height = xmax-xmin+1, ymax-ymin+1
    channels = []
    chunk = attrs["channels"][1]
    position = 0
    while chunk[position]:
        end = chunk.index(0, position)
        name = chunk[position:end].decode("ascii")
        position = end + 1
        typ = struct.unpack_from("<i", chunk, position)[0]
        channels.append((name, typ))
        position += 16
    pixels = {name: [0.0]*(width*height) for name, _ in channels}
    for i in range(height):
        start = struct.unpack_from("<Q", data, offset+i*8)[0]
        y, size = struct.unpack_from("<iI", data, start)
        pos = start + 8
        end = pos + size
        for name, typ in channels:
            code, bytes_per_sample = ("e", 2) if typ == 1 else ("f", 4)
            values = struct.unpack_from("<"+code*width, data, pos)
            pixels[name][(y-ymin)*width:(y-ymin+1)*width] = values
            pos += width*bytes_per_sample
        assert pos == end
    return width, height, channels, attrs, pixels

def inverse(value, curve):
    if curve == "srgb":
        return value/12.92 if value <= .04045 else ((value+.055)/1.055)**2.4
    if curve == "bt709":
        return value/4.5 if value < .081 else ((value+.099)/1.099)**(1/.45)
    return value

class FloatEXR(unittest.TestCase):
    cli = None

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-exr-reference-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.width, cls.height, cls.frames = 16, 8, 8
        cls.input = cls.root/"linear é 日本語.mkv"
        cls.samples = []
        raw = bytearray()
        for frame in range(cls.frames):
            values = {name: [] for name in "RGB"}
            for y in range(cls.height):
                for x in range(cls.width):
                    # Adjacent values below one RGB8 quantum plus curve branches.
                    for c, name in enumerate("RGB"):
                        values[name].append((1000 + x*13 + y*311 + frame*1231 + c*17000) % 65536)
            cls.samples.append(values)
            for name in "GBR":
                raw.extend(struct.pack("<"+"H"*(cls.width*cls.height), *values[name]))
        cls.raw = bytes(raw)
        run([FFMPEG, "-v", "error", "-f", "rawvideo", "-pixel_format", "gbrp16le", "-video_size", "16x8", "-framerate", "10", "-i", "pipe:0",
             "-vf", "setparams=color_primaries=bt709:color_trc=linear:colorspace=gbr:range=full", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp16le", "-color_trc", "linear", "-color_primaries", "bt709", "-colorspace", "rgb", "-color_range", "pc", cls.input], cls.raw)
        decoded = run([FFMPEG, "-v", "error", "-i", cls.input, "-f", "rawvideo", "-pix_fmt", "gbrp16le", "pipe:1"])
        assert decoded == cls.raw, "fixture must actually preserve all original 16-bit samples"
        probe = json.loads(run([FFPROBE,"-v","error","-show_streams","-of","json",cls.input]))["streams"][0]
        assert probe["color_transfer"] == "linear" and probe["color_primaries"] == "bt709"

    def invoke(self, *args, code=0):
        result = subprocess.run([str(self.cli), *map(str,args)],capture_output=True,timeout=60)
        self.assertEqual(result.returncode,code,result.stderr.decode("utf-8",errors="replace")+result.stdout.decode("utf-8",errors="replace"))
        return json.loads(result.stdout)

    def extract(self, directory, *flags):
        return self.invoke("extract",self.input,"--output",directory,"--interval","1","--format","exr","--exr-precision","float","--exr-compression","none","--quiet",*flags)

    def test_float_precision_and_metadata_independent_binary(self):
        output=self.root/"float \u00e9 \u65e5\u672c\u8a9e"
        actual=self.extract(output)
        self.assertEqual(actual["frames_accepted"],self.frames)
        for frame in range(self.frames):
            w,h,channels,attrs,pixels=uncompressed_exr(output/f"frame_{frame+1}.exr")
            self.assertEqual((w,h),(self.width,self.height))
            self.assertEqual(channels,[("B",2),("G",2),("R",2)])
            self.assertEqual(attrs["lfsOrigin"][1],b"decoded")
            self.assertEqual(attrs["lfsSourceComponentDepth"][1],b"16")
            self.assertIn("lfsSourceTimestamp",attrs)
            for name in "RGB":
                for expected,actual in zip(self.samples[frame][name],pixels[name]):
                    self.assertAlmostEqual(actual,expected/65535,delta=2e-6)
            self.assertGreater(len(set(pixels["R"][:16])),len(set(round(v*255) for v in pixels["R"][:16])))
        metadata=json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["output"]["color_profile"],"linear-sdr")
        self.assertEqual(metadata["output"]["bit_depth"],32)
        self.assertFalse(metadata["output"]["transfer_override"])
        self.assertEqual(metadata["output"]["input_transfer"],"linear")

    def test_half_zip_read_by_ffmpeg(self):
        output=self.root/"half-zip"
        self.invoke("extract",self.input,"--output",output,"--interval","1","--format","exr","--exr-precision","half","--end","0.01","--quiet")
        path=output/"frame_1.exr"
        native=json.loads(run([FFPROBE,"-v","error","-show_entries","stream=pix_fmt","-of","json",path]))["streams"][0]["pix_fmt"]
        self.assertIn(native,["gbrpf16le","gbrpf32le"],"independent reader must expose float samples without an integer display conversion")
        result=run([FFMPEG,"-v","error","-i",path,"-f","rawvideo","-pix_fmt",native,"pipe:1"])
        code,size=("e",2) if native=="gbrpf16le" else ("f",4)
        floats=struct.unpack("<"+code*(len(result)//size),result)
        for channel,name in enumerate("GBR"):
            for index,value in enumerate(self.samples[0][name]):
                reference=struct.unpack("<e",struct.pack("<e",value/65535))[0]
                self.assertAlmostEqual(floats[channel*self.width*self.height+index],reference,delta=1e-6)

    def test_transfers_and_linear_resize_rotation_reference(self):
        for curve in ["linear","srgb","bt709"]:
            for rotation in [0,90,180,270]:
                with self.subTest(curve=curve,rotation=rotation):
                    output=self.root/f"{curve}-{rotation}"
                    self.extract(output,"--input-transfer",curve,"--size","8","4","--rotate",rotation,"--end","0.01")
                    w,h,_,_,pixels=uncompressed_exr(output/"frame_1.exr")
                    self.assertEqual((w,h),(4,8) if rotation in [90,270] else (8,4))
                    for name in "RGB":
                        expected=[]
                        for y in range(4):
                            for x in range(8):
                                # Half scale pixel center: average four linearized samples.
                                expected.append(sum(inverse(self.samples[0][name][yy*16+xx]/65535,curve) for yy in [2*y,2*y+1] for xx in [2*x,2*x+1])/4)
                        rotated=[0.0]*32
                        for y in range(4):
                            for x in range(8):
                                dx,dy=(x,y) if rotation==0 else (3-y,x) if rotation==90 else (7-x,3-y) if rotation==180 else (y,7-x)
                                rotated[dy*w+dx]=expected[y*8+x]
                        for actual,reference in zip(pixels[name],rotated):
                            self.assertAlmostEqual(actual,reference,delta=3e-6)

    def test_fps_trim_window_and_timing(self):
        for flags in [["--fps","20"],["--fps","5","--start","0.1","--end","0.7"],["--fps","2","--window","--candidates","0"]]:
            with self.subTest(flags=flags):
                output=Path(tempfile.mkdtemp(dir=self.root))
                result=self.invoke("extract",self.input,"--output",output,"--format","exr","--exr-precision","float","--exr-compression","none","--quiet",*flags)
                metadata=json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(result["frames_accepted"],len(metadata["frames"]))
                for entry in metadata["frames"]:
                    _,_,_,_,pixels=uncompressed_exr(output/entry["file"])
                    index=entry["source_frame"]-1
                    self.assertAlmostEqual(pixels["R"][0],self.samples[index]["R"][0]/65535,delta=2e-6)

    def test_atomic_collision_manifest_failure_and_invalid_flags(self):
        output=self.root/"blocked-manifest";output.mkdir();(output/"extraction_metadata.json").write_bytes(b"keep")
        result=self.invoke("extract",self.input,"--output",output,"--format","exr","--end","0.01","--quiet",code=4)
        self.assertEqual(result["error"]["code"],"AlreadyExists")
        self.assertEqual((output/"extraction_metadata.json").read_bytes(),b"keep")
        self.assertTrue((output/"frame_1.exr").is_file())
        self.assertEqual(next(c["fields"]["frames_accepted"] for c in result["error"]["context"] if "frames_accepted" in c["fields"]),1)
        self.assertFalse(list(output.glob("*.tmp")))
        self.invoke("extract",self.input,"--output",output,"--format","exr","--end","0.01","--overwrite","--quiet")
        for flags in [["--exr-precision","bad"],["--exr-compression","bad"],["--input-transfer","pq"],["--input-primaries","bad"],["--format","png","--exr-precision","half"],["--format","exr","--quality","90"]]:
            destination=self.root/"invalid-exr"
            self.invoke("extract",self.input,"--output",destination,*flags,code=2)
            self.assertFalse(destination.exists())

    def test_unknown_hdr_and_yuv_colorimetry(self):
        # Deliberately remove source tags to exercise overrides and no assumptions.
        untagged=self.root/"untagged.nut"
        run([FFMPEG,"-v","error","-f","rawvideo","-pixel_format","gbrp16le","-video_size","16x8","-framerate","10","-i","pipe:0","-c:v","ffv1","-level","3",untagged],self.raw)
        output=self.root/"unknown-out"
        result=self.invoke("extract",untagged,"--output",output,"--format","exr","--quiet",code=3)
        self.assertEqual(result["error"]["code"],"Unsupported")
        self.assertFalse(output.exists())
        self.invoke("extract",untagged,"--output",output,"--format","exr","--input-transfer","linear","--input-primaries","bt709","--quiet")
        hdr=self.root/"pq.mkv"
        run([FFMPEG,"-v","error","-i",self.input,"-vf","setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=gbr:range=full","-c:v","ffv1","-level","3","-color_trc","smpte2084","-color_primaries","bt2020",hdr])
        result=self.invoke("extract",hdr,"--output",self.root/"hdr-out","--format","exr","--input-transfer","linear","--quiet",code=3)
        self.assertEqual(result["error"]["code"],"Unsupported")
        self.assertFalse((self.root/"hdr-out").exists())
        # Tagged YUV444 10-bit: verify conversion against FFmpeg's independent RGB48 output.
        yuv=self.root/"yuv10.mkv"
        run([FFMPEG,"-v","error","-i",self.input,"-vf","format=yuv444p10le,setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709:range=limited","-c:v","ffv1","-color_trc","linear","-color_primaries","bt709","-colorspace","bt709","-color_range","tv",yuv])
        output=self.root/"yuv-out"
        self.invoke("extract",yuv,"--output",output,"--format","exr","--exr-precision","float","--exr-compression","none","--end","0.01","--quiet")
        reference=run([FFMPEG,"-v","error","-i",yuv,"-frames:v","1","-vf","scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=bilinear+accurate_rnd","-f","rawvideo","-pix_fmt","rgb48le","pipe:1"])
        reference=struct.unpack("<"+"H"*(len(reference)//2),reference)
        _,_,_,_,pixels=uncompressed_exr(output/"frame_1.exr")
        for channel,name in enumerate("RGB"):
            for i,value in enumerate(pixels[name]):
                # FFmpeg RGB48 is an integer display reference and clips out-of-gamut samples.
                # This secondary display comparison allows one RGB8 level for its integer bridge.
                # Precision is qualified separately against analytic unclipped YUV at 2e-6.
                self.assertAlmostEqual(max(0,min(1,value)),reference[i*3+channel]/65535,delta=1/255)

    def test_yuv_nominal_range_superwhites_and_matrix_analytic(self):
        # No RGB intermediate in the fixture: encoded Y/Cb/Cr are the reference.
        count=16*8
        yy=[1023,0,940,64,500,500,500,500]*16
        cb=[512,512,512,512,64,960,512,512]*16
        cr=[512,512,512,512,512,512,64,960]*16
        raw=struct.pack("<"+"H"*(3*count),*(yy+cb+cr))
        for matrix,kr,kb,primaries in [("bt709",.2126,.0722,"bt709"),("smpte170m",.299,.114,"bt709"),("bt2020nc",.2627,.0593,"bt2020")]:
            for curve in ["linear","srgb","bt709"]:
                for full in [False,True]:
                    with self.subTest(matrix=matrix,curve=curve,full=full):
                        transfer="iec61966-2-1" if curve=="srgb" else curve
                        range_name="full" if full else "limited"
                        source=self.root/f"matrix-{matrix}-{curve}-{full}.mkv"
                        run([FFMPEG,"-v","error","-f","rawvideo","-pixel_format","yuv444p10le","-video_size","16x8","-framerate","10","-i","pipe:0",
                             "-vf",f"setparams=color_primaries={primaries}:color_trc={transfer}:colorspace={matrix}:range={range_name}","-c:v","ffv1","-level","3","-color_range","pc" if full else "tv",source],raw)
                        decoded=run([FFMPEG,"-v","error","-i",source,"-f","rawvideo","-pix_fmt","yuv444p10le","-color_range","pc" if full else "tv","pipe:1"])
                        self.assertEqual(decoded,raw,"fixture preserves all source YUV codes")
                        output=self.root/f"matrix-out-{matrix}-{curve}-{full}"
                        self.invoke("extract",source,"--output",output,"--format","exr","--exr-precision","float","--exr-compression","none","--quiet")
                        _,_,_,_,pixels=uncompressed_exr(output/"frame_1.exr")
                        for i in range(count):
                            y=yy[i]/1023 if full else (yy[i]-64)/876
                            b=(cb[i]-512)/(1023 if full else 896)
                            r=(cr[i]-512)/(1023 if full else 896)
                            encoded=[y+2*(1-kr)*r,y-2*kb*(1-kb)/(1-kr-kb)*b-2*kr*(1-kr)/(1-kr-kb)*r,y+2*(1-kb)*b]
                            for name,value in zip("RGB",encoded):
                                self.assertAlmostEqual(pixels[name][i],inverse(value,curve),delta=2e-6)
                        if not full:
                            self.assertGreater(pixels["R"][0],1)
                            self.assertLess(pixels["R"][1],0)

    def test_yuv_depths_and_subsampled_reference(self):
        # Raw planar codes are independent of either implementation's RGB conversion.
        for depth in [8,12,16]:
            with self.subTest(depth=depth):
                scale=1<<(depth-8)
                yy=[0,16*scale,235*scale,(1<<depth)-1]*32
                cb=[128*scale]*128
                cr=[128*scale]*128
                fmt="yuv444p" if depth==8 else f"yuv444p{depth}le"
                raw=bytes(yy+cb+cr) if depth==8 else struct.pack("<"+"H"*384,*(yy+cb+cr))
                source=self.root/f"depth-{depth}.mkv"
                run([FFMPEG,"-v","error","-f","rawvideo","-pixel_format",fmt,"-video_size","16x8","-framerate","10","-i","pipe:0",
                     "-vf","setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709:range=limited","-c:v","ffv1","-level","3","-color_range","tv",source],raw)
                self.assertEqual(run([FFMPEG,"-v","error","-i",source,"-f","rawvideo","-pix_fmt",fmt,"pipe:1"]),raw)
                output=self.root/f"depth-out-{depth}"
                self.invoke("extract",source,"--output",output,"--format","exr","--exr-precision","float","--exr-compression","none","--quiet")
                _,_,_,attrs,pixels=uncompressed_exr(output/"frame_1.exr")
                self.assertEqual(attrs["lfsSourceComponentDepth"][1],str(depth).encode())
                for name in "RGB":
                    for code,value in zip(yy,pixels[name]):
                        self.assertAlmostEqual(value,(code-16*scale)/(219*scale),delta=2e-6)
        # Native H.264/HEVC and raw-code references exercise actual subsampling.
        for depth in [8, 10]:
            for chroma_h in [1, 2]:
                for siting in ["left", "center", "topleft"]:
                    with self.subTest(depth=depth,chroma_h=chroma_h,siting=siting):
                        scale=1<<(depth-8)
                        yy=[(16+((i*19)%220))*scale for i in range(128)]
                        cb=[(40+(i*23)%190)*scale for i in range(64//chroma_h)]
                        cr=[(40+(i*31)%190)*scale for i in range(64//chroma_h)]
                        fmt=f"yuv4{2 if chroma_h==1 else 2}{2 if chroma_h==1 else 0}p"+("" if depth==8 else "10le")
                        values=yy+cb+cr
                        raw=bytes(values) if depth==8 else struct.pack("<"+"H"*len(values),*values)
                        source=self.root/f"sub-{depth}-{chroma_h}-{siting}.mkv"
                        run([FFMPEG,"-v","error","-f","rawvideo","-pixel_format",fmt,"-video_size","16x8","-framerate","10","-i","pipe:0",
                             "-vf","setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=limited","-c:v","ffv1","-level","3","-chroma_sample_location",siting,"-color_range","tv",source],raw)
                        self.assertEqual(run([FFMPEG,"-v","error","-i",source,"-f","rawvideo","-pix_fmt",fmt,"pipe:1"]),raw)
                        output=self.root/f"sub-out-{depth}-{chroma_h}-{siting}"
                        self.invoke("extract",source,"--output",output,"--format","exr","--exr-precision","float","--exr-compression","none","--quiet")
                        _,_,_,_,pixels=uncompressed_exr(output/"frame_1.exr")
                        def chroma(codes,x,y):
                            cx=x/2+(0 if siting in ["left","topleft"] else -.25)
                            cy=y/chroma_h+(0 if chroma_h==1 or siting=="topleft" else -.25)
                            ix,iy=int(cx//1),int(cy//1);fx,fy=cx-ix,cy-iy
                            def code(a,b): return codes[max(0,min(8//chroma_h-1,b))*8+max(0,min(7,a))]
                            return (code(ix,iy)*(1-fx)+code(ix+1,iy)*fx)*(1-fy)+(code(ix,iy+1)*(1-fx)+code(ix+1,iy+1)*fx)*fy
                        for y in range(8):
                            for x in range(16):
                                i=y*16+x
                                l=(yy[i]-16*scale)/(219*scale);b=(chroma(cb,x,y)-128*scale)/(224*scale);c=(chroma(cr,x,y)-128*scale)/(224*scale)
                                encoded=[l+2*(1-.2126)*c,l-2*.0722*(1-.0722)/.7152*b-2*.2126*(1-.2126)/.7152*c,l+2*(1-.0722)*b]
                                for name,value in zip("RGB",encoded): self.assertAlmostEqual(pixels[name][i],inverse(value,"bt709"),delta=3e-6)
        for codec,fmt in [("libx264","yuv420p"),("libx265","yuv420p10le")]:
            source=self.root/f"native-{codec}.mp4"
            run([FFMPEG,"-v","error","-i",self.input,"-vf","scale=64:64","-c:v",codec,"-pix_fmt",fmt,"-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709","-color_range","tv",source])
            result=self.invoke("extract",source,"--output",self.root/f"native-out-{codec}","--format","exr","--quiet")
            self.assertGreater(result["frames_accepted"],0)

    def test_area_reduction_removes_checkerboard_aliasing(self):
        # Every 4x4 block contains equal black/white pixels: the correct linear
        # area average is 0.5; a bilinear four-sample reduction aliases this pattern.
        source=self.root/"checker.mkv"
        codes=[65535 if (x%4==0 or x%4==3) else 0 for y in range(8) for x in range(16)]
        raw=struct.pack("<"+"H"*384,*(codes*3))
        run([FFMPEG,"-v","error","-f","rawvideo","-pixel_format","gbrp16le","-video_size","16x8","-framerate","10","-i","pipe:0",
             "-vf","setparams=color_primaries=bt709:color_trc=linear:colorspace=gbr:range=full","-c:v","ffv1","-level","3",source],raw)
        output=self.root/"checker-out"
        self.invoke("extract",source,"--output",output,"--format","exr","--exr-precision","float","--exr-compression","none","--scale","0.25","--quiet")
        w,h,_,_,pixels=uncompressed_exr(output/"frame_1.exr")
        self.assertEqual((w,h),(4,2))
        for name in "RGB":
            for value in pixels[name]: self.assertAlmostEqual(value,.5,delta=1e-6)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    FloatEXR.cli = args.cli
    unittest.main(argv=[__file__, *remaining])
