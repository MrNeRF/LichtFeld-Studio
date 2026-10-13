# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate synthetic HEVC fixtures with pillow-heif 1.8.1 (not needed by CI)."""
from pathlib import Path
import json,struct,hashlib
import pillow_heif
from PIL import Image, ImageCms
root=Path(__file__).resolve().parent
manifest={}
def stable_icc(profile):
 data=bytearray(profile)
 data[24:36]=struct.pack('>6H',2026,1,1,0,0,0)
 data[84:100]=bytes(16)
 return bytes(data)
for name,mode,bits,transfer,primaries,orientation in [
 ('yuv420','RGB',8,13,1,1),('bt709','RGB',8,1,1,1),('linear','RGB',8,8,1,1),('untagged','L',8,13,1,1),('hlg','RGB;16',12,18,9,1),('rgba12','RGBA;16',12,13,1,1),('rgb8','RGB',8,13,1,1),('mirrored','RGB',8,13,1,2),('premultiplied','RGBa',8,13,1,1),('rgba8','RGBA',8,13,1,1),
 ('rgb10','RGB;16',10,13,1,1),('rgb12','RGB;16',12,13,1,1),('p3','RGB',8,13,12,1),
 ('p3icc','RGB',8,13,1,1),('icc','RGB',8,13,1,1),('badicc','RGB',8,13,1,1),('multi','RGB',8,13,1,1),('pq','RGB;16',12,16,9,1),('rotated','RGB',8,13,1,6)]:
 channels=4 if mode.split(';')[0] in ('RGBA','RGBa') else 3
 width,height=(128,96) if name=="rgb8" else (32,24)
 values=[]
 for y in range(height):
  for x in range(width):
   values.extend([(32+x*5)%256,(40+y*7)%256,(60+x+y)%256]+([x*8] if channels==4 else []))
 if mode=='RGBa':
  for i in range(0,len(values),4):
   for c in range(3):values[i+c]=(values[i+c]*values[i+3]+127)//255
 # Native adjacent codes share the same RGB8 bucket. They must survive decode.
 if bits>8:
  native=[((1<<bits)//4+(i%31)) for i in range(len(values))]
  data=struct.pack('<'+'H'*len(native),*(v<<(16-bits) for v in native))
 else:data=bytes(values[::3]) if mode=="L" else bytes(values)
 image=pillow_heif.from_bytes(mode,(width,height),data)
 exif=Image.Exif();exif[274]=orientation
 if name=='multi':
  image.add_frombytes('RGB',(16,16),bytes([200,80,30])*256)
 extra={}
 if name=='rgb8':extra['tile_size']=64
 if name=='p3icc':extra['icc_profile']=(root/'display-p3.icc').read_bytes()
 if name=='icc':extra['icc_profile']=stable_icc(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
 if name=='badicc':extra['icc_profile']=b'not an ICC profile'
 if name=='multi':extra['primary_index']=1
 image.save(root/(name+'.heic'),quality=80 if name=='yuv420' else -1,chroma=420 if name=='yuv420' else 444,save_nclx_profile=name!='untagged',
            color_primaries=primaries,transfer_characteristics=transfer,matrix_coefficients=1 if name=='yuv420' else 0,
            bit_depth=bits,full_range_flag=0 if name=="yuv420" else 1,exif=exif.tobytes(),**extra,
            nclx_profile=dict(color_primaries=primaries,transfer_characteristics=transfer,matrix_coefficients=1 if name=='yuv420' else 0,full_range_flag=0 if name=='yuv420' else 1))
 decoded=pillow_heif.open_heif(root/(name+'.heic'),convert_hdr_to_8bit=False)
 raw=decoded.to_pillow().convert("RGB").tobytes() if mode=="L" else bytes(decoded.data)
 (root/(name+'.reference')).write_bytes(raw)
 manifest[name]={'size':decoded.size,'mode':decoded.mode,'stride':decoded.stride,'info':{k:v for k,v in decoded.info.items() if k in ('bit_depth','nclx_profile')},'sha256':hashlib.sha256((root/(name+'.heic')).read_bytes()).hexdigest()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
