# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate synthetic AV1 fixtures with pillow-heif 0.18.0 (not needed by CI)."""
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
def add_fixture_property(path, property_bytes=None, essential=False):
 # Pillow/libheif suppresses the second colr box when ICC is present.
 # Append one property to these authored, single-image fixtures without
 # re-encoding pixels. All box sizes and absolute item offsets are adjusted.
 data=bytearray(path.read_bytes())
 def boxes(start,end):
  result={}
  while start<end:
   size,kind=struct.unpack_from('>I4s',data,start)
   assert size>=8 and start+size<=end and kind not in result
   result[kind]=(start,size)
   start+=size
  assert start==end
  return result
 top=boxes(0,len(data));meta,meta_size=top[b'meta']
 children=boxes(meta+12,meta+meta_size)
 iprp,iprp_size=children[b'iprp']; props=boxes(iprp+8,iprp+iprp_size)
 ipco,ipco_size=props[b'ipco'];ipma,ipma_size=props[b'ipma']
 # Property types may repeat; count by their sizes instead of their names.
 p=ipco+8;count=0
 while p<ipco+ipco_size:
  size=struct.unpack_from('>I',data,p)[0];assert size>=8
  p+=size;count+=1
 assert p==ipco+ipco_size and count<127
 pitm,_=children[b'pitm'];primary=struct.unpack_from('>H',data,pitm+12)[0]
 assert data[ipma+8:ipma+12]==bytes(4)
 assert struct.unpack_from('>I',data,ipma+12)[0]==1
 assert struct.unpack_from('>H',data,ipma+16)[0]==primary
 associations=data[ipma+18];assoc_end=ipma+19+associations
 assert assoc_end==ipma+ipma_size and associations<255
 nclx=property_bytes or struct.pack('>I4s4sHHHB',19,b'colr',b'nclx',1,13,0,128)
 delta=len(nclx)+1
 iloc,iloc_size=children[b'iloc']
 assert data[iloc+8:iloc+14]==bytes(4)+b'\x44\x40'
 items=struct.unpack_from('>H',data,iloc+14)[0];p=iloc+16
 for _ in range(items):
  assert struct.unpack_from('>H',data,p+2)[0]==0
  base=struct.unpack_from('>I',data,p+4)[0]
  assert base>=meta+meta_size
  struct.pack_into('>I',data,p+4,base+delta)
  extents=struct.unpack_from('>H',data,p+8)[0];p+=10+extents*8
 assert p==iloc+iloc_size
 for pos,size,extra in [(meta,meta_size,delta),(iprp,iprp_size,delta),(ipco,ipco_size,len(nclx)),(ipma,ipma_size,1)]:
  struct.pack_into('>I',data,pos,size+extra)
 data[ipma+18]=associations+1
 for pos,extra in sorted([(ipco+ipco_size,nclx),(assoc_end,bytes([(count+1)|(128 if essential else 0)]))],reverse=True):
  data[pos:pos]=extra
 path.write_bytes(data)
def make_fixture_grid(path):
 # A 2x2 AV1 grid reuses an original encoded 64x64 tile, cropping its
 # bottom row to 128x96. This helper runs only during fixture generation.
 data=path.read_bytes()
 def box(kind,payload):return struct.pack('>I4s',len(payload)+8,kind)+payload
 def full(kind,version,payload):return box(kind,bytes([version,0,0,0])+payload)
 def children(start,end):
  result={}
  while start<end:
   size,kind=struct.unpack_from('>I4s',data,start)
   assert size>=8 and start+size<=end and kind not in result
   result[kind]=(start,size);start+=size
  assert start==end
  return result
 def raw(entry):return data[entry[0]:sum(entry)]
 top=children(0,len(data));meta,meta_size=top[b'meta']
 old=children(meta+12,meta+meta_size);iprp,iprp_size=old[b'iprp']
 ipco,ipco_size=children(iprp+8,iprp+iprp_size)[b'ipco']
 properties=children(ipco+8,ipco+ipco_size)
 iloc,_=old[b'iloc'];assert data[iloc+8:iloc+14]==bytes(4)+b'\x44\x40'
 assert struct.unpack_from('>H',data,iloc+16)[0]==1
 base=struct.unpack_from('>I',data,iloc+20)[0]
 assert struct.unpack_from('>H',data,iloc+24)[0]==1
 offset,length=struct.unpack_from('>II',data,iloc+26)
 tile=data[base+offset:base+offset+length];assert len(tile)==length
 grid=struct.pack('>BBBBHH',0,0,1,1,128,96)
 prop=box(b'ipco',raw(properties[b'av1C'])+full(b'ispe',0,struct.pack('>II',64,64))+full(b'ispe',0,struct.pack('>II',128,96))+raw(properties[b'pixi'])+raw(properties[b'colr']))
 assoc=full(b'ipma',0,struct.pack('>I',5)+b''.join(struct.pack('>HB4B',i,4,129,130,4,5) for i in range(1,5))+struct.pack('>HB3B',5,3,131,4,5))
 info=full(b'iinf',0,struct.pack('>H',5)+b''.join(full(b'infe',2,struct.pack('>HH4sB',i,0,b'av01',0)) for i in range(1,5))+full(b'infe',2,struct.pack('>HH4sB',5,0,b'grid',0)))
 references=full(b'iref',0,box(b'dimg',struct.pack('>HH4H',5,4,1,2,3,4)))
 def metadata(base):
  location=full(b'iloc',0,b'\x44\x00'+struct.pack('>H',5)+b''.join(struct.pack('>HHHII',i,0,1,base+(i-1)*len(tile),len(tile)) for i in range(1,5))+struct.pack('>HHHII',5,0,1,base+4*len(tile),len(grid)))
  return full(b'meta',0,raw(old[b'hdlr'])+full(b'pitm',0,struct.pack('>H',5))+location+info+box(b'iprp',prop+assoc)+references)
 ftyp=raw(top[b'ftyp']);header=metadata(0)
 header=metadata(len(ftyp)+len(header)+8)
 path.write_bytes(ftyp+header+box(b'mdat',tile*4+grid))
for name,mode,bits,transfer,primaries,orientation in [
 ('yuv420','RGB',8,13,1,1),('bt709','RGB',8,1,1,1),('linear','RGB',8,8,1,1),('untagged','L',8,13,1,1),('hlg','RGB;16',12,18,9,1),('rgba12','RGBA;16',12,13,1,1),('rgb8','RGB',8,13,1,1),('mirrored','RGB',8,13,1,2),('premultiplied','RGBa',8,13,1,1),('rgba8','RGBA',8,13,1,1),
 ('rgb10','RGB;16',10,13,1,1),('rgb12','RGB;16',12,13,1,1),('p3','RGB',8,13,12,1),
 ('p3icc','RGB',8,13,1,1),('icc','RGB',8,13,1,1),('badicc','RGB',8,13,1,1),('multi','RGB',8,13,1,1),('pq','RGB;16',12,16,9,1),('rotated','RGB',8,13,1,6)]:
 channels=4 if mode.split(';')[0] in ('RGBA','RGBa') else 3
 width,height=(64,64) if name=="rgb8" else (32,24)
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
 if name=='p3icc':extra['icc_profile']=(root/'display-p3.icc').read_bytes()
 if name=='icc':extra['icc_profile']=stable_icc(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
 if name=='badicc':extra['icc_profile']=b'not an ICC profile'
 if name=='multi':extra['primary_index']=1
 image.save(root/(name+'.avif'),format='AVIF',quality=80 if name=='yuv420' else -1,chroma=420 if name=='yuv420' else 444,save_nclx_profile=name!='untagged',
            color_primaries=primaries,transfer_characteristics=transfer,matrix_coefficients=1 if name=='yuv420' else 0,
            bit_depth=bits,full_range_flag=0 if name=="yuv420" else 1,exif=exif.tobytes(),**extra,
            nclx_profile=dict(color_primaries=primaries,transfer_characteristics=transfer,matrix_coefficients=1 if name=='yuv420' else 0,full_range_flag=0 if name=='yuv420' else 1))
 if name in ('p3icc','badicc'):add_fixture_property(root/(name+'.avif'))
 if name=='rotated':add_fixture_property(root/(name+'.avif'),struct.pack('>I4sB',9,b'irot',3),True)
 if name=='mirrored':add_fixture_property(root/(name+'.avif'),struct.pack('>I4sB',9,b'imir',1),True)
 if name=='rgb8':make_fixture_grid(root/(name+'.avif'))
 decoded=pillow_heif.open_heif(root/(name+'.avif'),convert_hdr_to_8bit=False)
 raw=decoded.to_pillow().convert("RGB").tobytes() if mode=="L" else bytes(decoded.data)
 (root/(name+'.reference')).write_bytes(raw)
 manifest[name]={'size':decoded.size,'mode':decoded.mode,'stride':decoded.stride,'info':{k:v for k,v in decoded.info.items() if k in ('bit_depth','nclx_profile')},'sha256':hashlib.sha256((root/(name+'.avif')).read_bytes()).hexdigest()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
