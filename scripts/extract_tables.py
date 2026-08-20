"""Extract atomic scattering coefficient tables from EMsoft Fortran sources
into a Python data module for torchebsd. Faithful, no manual transcription.

- Weickenmeier-Kohl AA(4,98), BB(4,98) from mod_others.f90 (GETWK)
- Doyle-Turner/Smith-Burge scatfac(8,98) from mod_diffraction.f90
- element symbols SY(98)
"""
import re
from pathlib import Path

OTHERS = Path(r"D:/emsoft/EMsoftOO/Source/EMsoftOOLib/mod_others.f90")
DIFF = Path(r"D:/emsoft/EMsoftOO/Source/EMsoftOOLib/mod_diffraction.f90")


def grab_reshape(text, varname, ncols_rows):
    """Grab the numbers inside a `varname ... reshape((/ ... /),(/r,c/))` block."""
    # find 'reshape( (/' or 'reshape((/' after varname
    i = text.index(varname)
    m = re.search(r"reshape\(\s*\(/", text[i:])
    start = i + m.end()
    k = text.index("/)", start)
    body = text[start:k]
    nums = re.findall(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?|[-+]?\d+\.\d*", body)
    nums = [float(n.replace("D", "E").replace("d", "e")) for n in nums]
    return nums


def grab_symbols(text):
    i = text.index("SY(98)")
    j = text.index("(/", i)
    k = text.index("/)", j)
    body = text[j + 2:k]
    syms = re.findall(r"'([A-Za-z ]{1,2})'", body)
    return [s.strip() for s in syms]


others = OTHERS.read_text(encoding="utf-8", errors="ignore")
diff = DIFF.read_text(encoding="utf-8", errors="ignore")

AA = grab_reshape(others, "AA(4,98)", (4, 98))
BB = grab_reshape(others, "BB(4,98)", (4, 98))
syms = grab_symbols(others)
scatfac = grab_reshape(diff, "scatfac(8,98)", (8, 98))

assert len(AA) == 4 * 98, len(AA)
assert len(BB) == 4 * 98, len(BB)
assert len(syms) == 98, len(syms)
assert len(scatfac) == 8 * 98, len(scatfac)

# Fortran reshape is column-major: reshape((/...),(/4,98/)) -> element(r,c)
# stored so first 4 entries are column 1 (=atom 1). So atom z uses entries [4z:4z+4].
AA_rows = [AA[4 * z:4 * z + 4] for z in range(98)]  # AA_rows[z-1] = [a1..a4] for element z
BB_rows = [BB[4 * z:4 * z + 4] for z in range(98)]
SC_rows = [scatfac[8 * z:8 * z + 8] for z in range(98)]  # [a1..a4,b1..b4]

out = Path(__file__).resolve().parents[1] / "src/torchebsd/data/scattering_tables.py"
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    f.write('"""Atomic scattering-factor coefficient tables (auto-generated).\n\n')
    f.write("Extracted verbatim from EMsoft (EMsoftOO) Fortran sources:\n")
    f.write("  - WK_A, WK_B: Weickenmeier-Kohl 4-Gaussian coefficients, from\n")
    f.write("    mod_others.f90 GETWK subroutine (AA(4,98), BB(4,98)).\n")
    f.write("  - DT: Doyle-Turner/Smith-Burge 4-Gaussian (a1..a4,b1..b4) from\n")
    f.write("    mod_diffraction.f90 scatfac(8,98).\n")
    f.write("Row index i corresponds to atomic number Z = i + 1 (Z in 1..98).\n")
    f.write("Do not edit by hand; regenerate with scripts/extract_tables.py.\n")
    f.write('"""\n\n')
    f.write("ELEMENT_SYMBOLS = [\n")
    for z in range(0, 98, 10):
        f.write("    " + ", ".join(repr(s) for s in syms[z:z + 10]) + ",\n")
    f.write("]\n\n")

    def dump(name, rows, ncol):
        f.write(f"{name} = [\n")
        for r in rows:
            f.write("    [" + ", ".join(repr(v) for v in r) + "],\n")
        f.write("]\n\n")

    dump("WK_A", AA_rows, 4)
    dump("WK_B", BB_rows, 4)
    dump("DT_AB", SC_rows, 8)

print("wrote", out)
print("Si (Z=14) WK A:", AA_rows[13], "B:", BB_rows[13], "sym:", syms[13])
print("Ni (Z=28) WK A:", AA_rows[27], "sym:", syms[27])
