# GoPro GPMF fixture

Source: https://github.com/gopro/gpmf-parser/blob/a9445b4bf46ac7d66bb3bad54b6abfd8e68ad4be/samples/hero6a.mp4
Revision: a9445b4bf46ac7d66bb3bad54b6abfd8e68ad4be
License: MIT (upstream LICENSE-MIT, copied as LICENSE).

`hero6a-gpmd.mp4` contains only stream 3 of the original. It was remuxed with:

```
ffmpeg -i hero6a.mp4 -map 0:3 -c copy -copy_unknown -tag:d gpmd hero6a-gpmd.mp4
```

Original SHA-256: cb7cdbe9be1122f6f9f25019377e1eacc0ef671d6836c13ab934c9b8f5c22cad
Fixture SHA-256: 57afbdbeeb849fad81925c8534e7329b7e7c362ca9bb8b1a3d3bfc9ed3f99a01
