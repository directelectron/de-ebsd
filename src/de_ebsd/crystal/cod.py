"""Finding a structure in the Crystallography Open Database (COD).

COD matches elements EXACTLY, so a search is by the set of elements a phase is made
of (:func:`elements` reads them however they are typed), and :func:`search` answers
one row per structure, the most-studied first. :func:`fetch` downloads an entry's
CIF, which :meth:`Phase.from_cif <de_ebsd.crystal.Phase.from_cif>` reads.

Only the standard library speaks to the network, so this needs no extra; it does
need the network, and raises when COD cannot be reached.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

__all__ = ["COD", "elements", "fetch", "search", "symmetry"]

log = logging.getLogger(__name__)

COD = "https://www.crystallography.net/cod"
#: Seconds a request may take: a search is given up when COD is down.
TIMEOUT = 20
#: Rows :func:`search` keeps.
MAX_RESULTS = 40
#: Seconds between tries when COD drops the connection, which it does now and then
#: under load. A timeout is not tried again: COD is down, not busy.
RETRY_S = (0.5, 1.5)
SYMBOLS = frozenset(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au
    Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf
    Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split()
)


def _agent() -> dict[str, str]:
    from de_ebsd import __version__

    return {"User-Agent": f"de_ebsd/{__version__}"}


def elements(text: str) -> list[str] | None:
    """The element symbols typed in *text* -- ``"Fe Cr"``, ``"Fe-Cr"``, ``"FeCr"`` or
    ``"fe cr"`` -- each once, or ``None`` if any part is not one, so a half-typed
    symbol is not searched for as some other element."""
    found = []
    for word in re.split(r"[^A-Za-z]+", text):
        if not word:
            continue
        # capitals split a run of symbols; a word without any is one symbol
        parts = re.findall(r"[A-Z][a-z]?", word) if word[0].isupper() else [word]
        if "".join(parts) != word:
            return None
        for part in parts:
            symbol = part.capitalize()
            if symbol not in SYMBOLS:
                return None
            if symbol not in found:
                found.append(symbol)
    return found


@functools.cache
def symmetry(symbol: str) -> tuple[str, str]:
    """The point group and crystal system of the space group *symbol* as COD writes
    it (``"F m -3 m"``, ``"R -3 m :H"``): ``("m-3m", "cubic")``, or empty strings
    for a symbol that cannot be read. The formula alone hides how different two
    entries are: hexagonal and cubic nickel are both "Ni"."""
    from diffpy.structure import spacegroups
    from orix.quaternion.symmetry import get_point_group

    # newer diffpy names it get_space_group and deprecates GetSpaceGroup
    lookup = getattr(spacegroups, "get_space_group", None) or spacegroups.GetSpaceGroup
    try:
        group = lookup(symbol.split(":")[0].strip())
        point_group = get_point_group(group.number, proper=False).name
    except Exception:  # noqa: BLE001 - diffpy raises a variety on a symbol it cannot read
        return "", ""
    return point_group, str(group.crystal_system).lower()


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read(url: str) -> str:
    """The body at *url*, tried again when the connection is dropped."""
    request = urllib.request.Request(url, headers=_agent())
    for wait in (*RETRY_S, None):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read().decode("utf-8", "replace")
        except (ConnectionError, urllib.error.URLError) as e:
            dropped = isinstance(e, ConnectionError) or isinstance(
                getattr(e, "reason", None), ConnectionError
            )
            if not dropped or wait is None:
                raise
            log.debug("COD dropped the connection (%s); trying again", e)
            time.sleep(wait)
    raise AssertionError("unreachable")


def search(symbols) -> list[dict]:
    """Structures made of exactly the elements *symbols* (``["Fe", "Cr"]``).

    One row per structure (formula and space group), each a dict of ``id, formula,
    phase, sg, point_group, system, entries, a, b, c, alpha, beta, gamma, volume``
    (lengths in Angstrom, as COD gives them). A row's cell is the determination
    nearest the median volume of all of them, not an outlier at high pressure or
    temperature, and a named entry beats an unnamed one of the same cell.

    The most-studied structures come first: ``entries`` is how many of COD's entries
    the row stands for, and a common phase has been determined over and over (FCC
    nickel 98 times, the hexagonal form a few). At most :data:`MAX_RESULTS` rows.
    Raises on a network error.
    """
    symbols = [str(e).strip().capitalize() for e in symbols if str(e).strip()]
    if not symbols:
        return []
    n = len(symbols)
    params = [(f"el{i + 1}", e) for i, e in enumerate(symbols)]
    params += [("strictmin", str(n)), ("strictmax", str(n)), ("format", "json")]
    text = _read(f"{COD}/result.php?" + urllib.parse.urlencode(params))
    try:
        raw = json.loads(text) if text.strip() else []
    except json.JSONDecodeError:
        return []
    found: dict[tuple, list[dict]] = {}
    for r in raw:
        a, b, c = _number(r.get("a")), _number(r.get("b")), _number(r.get("c"))
        if a is None or b is None or c is None:
            continue
        formula = (r.get("formula") or r.get("calcformula") or "").strip(" -") or "?"
        sg = (r.get("sg") or "").strip()
        name = r.get("mineral") or r.get("commonname") or r.get("chemname") or ""
        found.setdefault((formula, sg), []).append(
            {
                "id": str(r.get("file", "")),
                "formula": formula,
                "phase": name.strip(),
                "sg": sg,
                "a": a,
                "b": b,
                "c": c,
                "alpha": _number(r.get("alpha")),
                "beta": _number(r.get("beta")),
                "gamma": _number(r.get("gamma")),
                "volume": _number(r.get("vol")),
            }
        )
    rows = []
    for (_, sg), determinations in found.items():
        volumes = sorted(d["volume"] for d in determinations if d["volume"] is not None)
        median = volumes[len(volumes) // 2] if volumes else None

        def spread(d, median=median):
            if median is None or d["volume"] is None:
                return float("inf")
            return abs(d["volume"] - median)

        row = dict(min(determinations, key=lambda d: (round(spread(d), 1), not d["phase"])))
        row["point_group"], row["system"] = symmetry(sg)
        row["entries"] = len(determinations)
        rows.append(row)
    rows.sort(key=lambda e: (-e["entries"], e["volume"] or 1e9))
    return rows[:MAX_RESULTS]


def fetch(cod_id, directory=None) -> Path:
    """COD entry *cod_id* saved as ``cod_<id>.cif`` in *directory* (the temporary
    folder by default): its path. Raises ``ValueError`` for an id with no digits or
    a reply that is not a CIF."""
    cod_id = "".join(ch for ch in str(cod_id) if ch.isdigit())
    if not cod_id:
        raise ValueError("not a COD id")
    text = _read(f"{COD}/{cod_id}.cif")
    if "loop_" not in text and "_cell_length_a" not in text:
        raise ValueError("COD's reply was not a CIF")
    path = Path(tempfile.gettempdir() if directory is None else directory) / f"cod_{cod_id}.cif"
    path.write_text(text, encoding="utf-8")
    return path
