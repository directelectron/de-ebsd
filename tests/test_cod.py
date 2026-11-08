"""Searching the Crystallography Open Database, without the network: every reply here
is recorded, and handed to the module in place of COD's. Ported from Autopilot's
``tests/test_ebsd.py`` (its copy of SpyDE's COD search)."""

import io
import json
import urllib.parse

import pytest

from de_ebsd.crystal import cod


def test_the_elements_are_read_however_they_are_typed():
    for typed in ("Fe Cr", "FeCr", "fe, cr", "Fe-Cr"):
        assert cod.elements(typed) == ["Fe", "Cr"], typed
    assert cod.elements("ni") == ["Ni"]
    assert cod.elements("Fe fe") == ["Fe"]
    # not symbols, so not searched for as some other element
    assert cod.elements("Fx") is None and cod.elements("fecr") is None


def _entry(file, sg, a, c=None, name=""):
    c = a if c is None else c
    hexagonal = "63" in sg
    return {
        "file": file,
        "formula": "- Ni -",
        "sg": sg,
        "a": a,
        "b": a,
        "c": c,
        "vol": a * a * c * (0.866 if hexagonal else 1.0),
        "alpha": 90,
        "beta": 90,
        "gamma": 120 if hexagonal else 90,
        "mineral": name,
    }


def test_the_commonest_structure_comes_first_one_row_each(monkeypatch):
    """FCC nickel, determined over and over, before the hexagonal form COD has two
    of -- each one row, on its typical cell, with its point group."""
    fcc = [_entry(f"9{i}", "F m -3 m", 3.524 + 0.001 * i) for i in range(5)]
    fcc.append(_entry("950", "F m -3 m", 3.45))  # at high pressure: not the typical cell
    hexagonal = [_entry(f"8{i}", "P 63/m m c", 2.62, 4.31) for i in range(2)]
    hexagonal.append({**_entry("870", "P 63/m m c", 2.62), "a": None})  # no cell: skipped
    asked = []

    def read(url):
        asked.append(url)
        return json.dumps(hexagonal + fcc)

    monkeypatch.setattr(cod, "_read", read)
    rows = cod.search(["ni"])
    assert [(r["system"], r["point_group"], r["entries"]) for r in rows] == [
        ("cubic", "m-3m", 6),
        ("hexagonal", "6/mmm", 2),
    ]
    assert rows[0]["a"] == pytest.approx(3.526, abs=0.002)
    assert rows[0]["formula"] == "Ni"
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(asked[0]).query))
    assert query == {"el1": "Ni", "strictmin": "1", "strictmax": "1", "format": "json"}


def test_a_named_entry_beats_an_unnamed_one_of_the_same_cell(monkeypatch):
    entries = [_entry("1", "F m -3 m", 3.524), _entry("2", "F m -3 m", 3.524, name="nickel")]
    monkeypatch.setattr(cod, "_read", lambda url: json.dumps(entries))
    assert cod.search(["Ni"])[0]["phase"] == "nickel"


def test_nothing_to_search_or_nothing_found_is_no_rows(monkeypatch):
    monkeypatch.setattr(cod, "_read", lambda url: "")
    assert cod.search([]) == [] and cod.search([" "]) == []
    assert cod.search(["Ni"]) == []
    monkeypatch.setattr(cod, "_read", lambda url: "<html>busy</html>")
    assert cod.search(["Ni"]) == []


def test_a_dropped_connection_is_tried_again(monkeypatch):
    tries = []

    def urlopen(request, timeout):
        tries.append(request)
        if len(tries) == 1:
            raise ConnectionResetError(10054, "forcibly closed by the remote host")
        return io.BytesIO(b"[]")

    monkeypatch.setattr(cod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cod, "RETRY_S", (0.0, 0.0))
    assert cod.search(["Ni"]) == []
    assert len(tries) == 2
    assert tries[0].get_header("User-agent").startswith("de_ebsd/")


def test_a_timeout_is_not_tried_again(monkeypatch):
    tries = []

    def urlopen(request, timeout):
        tries.append(request)
        raise cod.urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(cod.urllib.request, "urlopen", urlopen)
    with pytest.raises(cod.urllib.error.URLError):
        cod.search(["Ni"])
    assert len(tries) == 1


def test_a_space_group_as_cod_writes_it_has_a_point_group():
    assert cod.symmetry("F m -3 m") == ("m-3m", "cubic")
    assert cod.symmetry("P 63/m m c") == ("6/mmm", "hexagonal")
    assert cod.symmetry("R -3 c :H") == ("-3m", "trigonal")  # a setting after the colon
    assert cod.symmetry("not a group") == ("", "")


CIF = """data_nickel
_cell_length_a 3.524
loop_
_atom_site_label
Ni1
"""


def test_an_entry_is_fetched_as_a_cif(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(cod, "_read", lambda url: asked.append(url) or CIF)
    path = cod.fetch("COD 9008476", directory=tmp_path)
    assert asked == [f"{cod.COD}/9008476.cif"]
    assert path == tmp_path / "cod_9008476.cif"
    assert path.read_text(encoding="utf-8") == CIF


def test_a_reply_that_is_not_a_cif_or_an_id_without_digits_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(cod, "_read", lambda url: "<html>not found</html>")
    with pytest.raises(ValueError, match="not a CIF"):
        cod.fetch(9008476, directory=tmp_path)
    with pytest.raises(ValueError, match="not a COD id"):
        cod.fetch("nickel")
