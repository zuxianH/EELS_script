import copy
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.units import Bohr

from tacaw_studio.config import defaults
from tacaw_studio.trajectories import (convert, discover, frames, preview, validate_beads,
                                       supported_cell, displayed_atoms)

def options():
    return defaults()["inputs"]["conversion"]

def test_discovery_preference_order_and_exclusions(tmp_path):
    for name in ("bead10.traj", "bead2.traj", "bead1.xyz"):
        (tmp_path/name).touch()
    for name in ("tacaw_runs", "out.zarray", "another_run"):
        d = tmp_path/name
        d.mkdir()
        (d/"hidden.traj").touch()
        if name == "another_run":
            (d/"studio.yaml").touch()
    sub = tmp_path/"inputs"
    sub.mkdir()
    (sub/"b.traj").touch()
    assert [p.name for p in discover(tmp_path)] == ["bead2.traj", "bead10.traj", "bead1.xyz"]
    assert "b.traj" in [p.name for p in discover(tmp_path, True)]
    assert "hidden.traj" not in [p.name for p in discover(tmp_path, True)]

def test_ipi_units_selection_and_metadata(tmp_path):
    source = tmp_path/"simulation.xc.xyz"
    source.write_text("".join(f"1\n# CELL(abcABC): 10 12 14 90 90 90 Step: {i} Bead: 0 x_centroid{{angstrom}} cell{{atomic_unit}}\nSi {1+i*.1} 2 3\n" for i in range(8)))
    c = options()
    c.update(start=1, stop=7, stride=2)
    target = tmp_path/"converted.traj"
    assert convert(str(source), target, c) == 3
    with Trajectory(target) as t:
        assert len(t) == 3
        np.testing.assert_allclose(t[0].cell.lengths(), np.array([10,12,14])*Bohr)
        assert t[0].positions[0,0] == pytest.approx(1.1)
        assert [a.info["step"] for a in t] == [1,3,5]
    assert not list(tmp_path.glob("*.partial.traj"))

def test_extended_xyz_respects_properties_order(tmp_path):
    path = tmp_path/"input.extxyz"
    path.write_text('1\nLattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:charge:R:1:pos:R:3 pbc="T T T"\nSi 42 1 2 3\n')
    atoms, count = preview(str(path), 0, options())
    np.testing.assert_allclose(atoms.positions, [[1,2,3]])
    assert count is None

def test_atomic_conversion_preserves_target_on_error(tmp_path):
    source = tmp_path/"bad.xyz"
    source.write_text("1\nCELL(abcABC): 8 8 8 90 90 90\nSi 1 2 3\n2\nmissing data\n")
    target = tmp_path/"out.traj"
    with pytest.raises(ValueError):
        convert(str(source), target, options())
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial.traj"))
    target.write_text("keep me")
    with pytest.raises(FileExistsError):
        convert(str(source), target, options())
    assert target.read_text() == "keep me"

def test_missing_cell_requires_override(tmp_path):
    source = tmp_path/"plain.xyz"
    source.write_text("1\nplain\nSi 1 2 3\n")
    with pytest.raises(ValueError, match="cell"):
        convert(str(source), tmp_path/"bad.traj", options())
    c = options()
    c.update(cell=[8,8,8], pbc=[True,True,True])
    assert convert(str(source), tmp_path/"good.traj", c) == 1

def test_stream_lammps_with_element_mapping(tmp_path):
    path = tmp_path/"trajectory.dump"
    path.write_text("".join(f"ITEM: TIMESTEP\n{i}\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 8\n0 8\n0 8\nITEM: ATOMS id type x y z\n1 1 1 2 3\n" for i in range(3)))
    c = options()
    c["type_symbols"] = {1:"Si"}
    atoms = list(frames(path, c))
    assert len(atoms) == 3
    assert all(a.get_chemical_symbols() == ["Si"] for a in atoms)

def test_bead_lengths_order_and_time(source, tmp_path):
    other = tmp_path/"other.traj"
    with Trajectory(source) as t, Trajectory(other,"w") as out:
        for a in t:
            out.write(a)
    assert validate_beads([str(source),str(other)])["beads"] == 2
    with Trajectory(source) as t, Trajectory(other,"w") as out:
        a = t[0]
        a.info["step"] = 99
        out.write(a)
    with pytest.raises(ValueError, match="counts"):
        validate_beads([str(source),str(other)])
    with Trajectory(source) as t, Trajectory(other,"w") as out:
        for a in t:
            a.info["step"] += 1
            out.write(a)
    with pytest.raises(ValueError, match="step"):
        validate_beads([str(source),str(other)])
    with Trajectory(source) as t, Trajectory(other,"w") as out:
        for a in t:
            a.numbers[0] = 8
            out.write(a)
    with pytest.raises(ValueError, match="ordering"):
        validate_beads([str(source),str(other)])

def test_cell_support_and_z_filter():
    atoms = Atoms("Si2", positions=[[1,1,1],[1,1,6]], cell=[8,8,8], pbc=True)
    supported_cell(atoms)
    positions, numbers, indices = displayed_atoms(atoms, 0,8, repeats=2)
    assert indices.tolist() == [0]
    atoms.cell[0,1] = 1
    with pytest.raises(ValueError, match="diagonal"):
        supported_cell(atoms)
