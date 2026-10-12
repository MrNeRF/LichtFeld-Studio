#!/usr/bin/env python3
"""Regenerate checked-in XKB keysyms; run with setxkbmap, xkbcomp, libxkbcommon."""
import argparse, ctypes, json, re, subprocess, tempfile
from pathlib import Path
OUTPUT = Path(__file__).resolve().parents[1] / 'tests/data/keymap_layouts'
POSITIONS = {'TLDE':53, 'AE11':45, 'AE12':46, 'AD11':47, 'AD12':48, 'AC10':51, 'AC11':52,
             'AB08':54, 'AB09':55, 'AB10':56, 'BKSL':49, 'LSGT':100, 'SPCE':44}
POSITIONS.update({f'AE{i:02d}':29+i for i in range(1,11)})
POSITIONS.update(dict(zip((f'AD{i:02d}' for i in range(1,11)), [20,26,8,21,23,28,24,12,18,19])))
POSITIONS.update(dict(zip((f'AC{i:02d}' for i in range(1,10)), [4,22,7,9,10,11,13,14,15])))
POSITIONS.update(dict(zip((f'AB{i:02d}' for i in range(1,8)), [29,27,6,25,5,17,16])))
LAYOUTS = [('us','us',''),('gb','gb',''),('de','de',''),('ch','ch',''),('fr','fr',''),('be','be',''),('es','es',''),('it','it',''),('pt','pt',''),('se','se',''),('no','no',''),('dk','dk',''),('pl','pl',''),('cz','cz',''),('us-dvorak','us','dvorak'),('us-colemak','us','colemak'),('ru','ru',''),('ua','ua',''),('jp','jp','')]
lib = ctypes.CDLL('libxkbcommon.so.0')
lib.xkb_keysym_from_name.argtypes = [ctypes.c_char_p, ctypes.c_int]; lib.xkb_keysym_from_name.restype = ctypes.c_uint32
lib.xkb_keysym_to_utf32.argtypes = [ctypes.c_uint32]; lib.xkb_keysym_to_utf32.restype = ctypes.c_uint32

def unicode_symbol(name):
    if name == 'NoSymbol': return 0
    return int(lib.xkb_keysym_to_utf32(lib.xkb_keysym_from_name(name.encode(), 0)))

def generate(label, layout, variant):
    cmd = ['setxkbmap','-rules','evdev','-model','pc105','-layout',layout]
    if variant: cmd += ['-variant',variant]
    spec = subprocess.check_output(cmd + ['-print'], text=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d); (p/'spec.xkb').write_text(spec)
        subprocess.run(['xkbcomp','-xkb',str(p/'spec.xkb'),str(p/'expanded.xkb')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        source = (p/'expanded.xkb').read_text()
    rows = []
    for match in re.finditer(r'key\s+<(\w+)>\s*\{(.*?)\};', source, re.S):
        pos, body = match.groups()
        if pos not in POSITIONS: continue
        sym = re.search(r'symbols\[Group1\]\s*=\s*\[([^]]+)\]', body) or re.search(r'\[([^]]+)\]', body)
        if not sym: continue
        names = [v.strip() for v in sym.group(1).split(',')]
        shifted = names[1] if len(names) > 1 else names[0]
        rows.append(dict(position=pos, scancode=POSITIONS[pos], unshifted=unicode_symbol(names[0]), shifted=unicode_symbol(shifted), unshifted_name=names[0], shifted_name=shifted))
    return dict(layout=label, xkb_layout=layout, xkb_variant=variant, rows=sorted(rows, key=lambda r:r['scancode']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        data = generate(*layout)
        if len(data['rows']) != len(POSITIONS): raise SystemExit(f'{layout[0]}: expected {len(POSITIONS)} positions, got {len(data["rows"])}')
        path = OUTPUT / f'{layout[0]}.json'; content = json.dumps(data, separators=(',',':'), ensure_ascii=False)+'\n'
        if args.check and (not path.exists() or path.read_text() != content): raise SystemExit(f'Stale layout table: {path}; run python3 tools/generate_keymap_layouts.py')
        if not args.check: path.write_text(content)
        print(f'{layout[0]}: {len(data["rows"])} keys')
