"""One-off: extract per-chain sequences from every .pdb in a directory.

Usage:
    python scripts/extract_sequences.py <pdb_dir> [<output_csv>]

Writes one row per (pdb, chain) with columns: pdb, chain, length, sequence.
Matches the residue-name -> 1-letter convention already used by
evaluations/protein_utils.py.

Defaults: output_csv = <pdb_dir>/sequences.csv.
"""

from __future__ import annotations

import csv
import glob
import os
import sys

from biotite.structure.io.pdb import PDBFile


# Mirrors evaluations/protein_utils.RESIDUE_TYPES_3to1.
_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def chain_sequences(pdb_path: str) -> list[tuple[str, str]]:
    """Return (chain_id, sequence) pairs for the first model of a PDB."""
    atoms = PDBFile.read(pdb_path).get_structure()[0]
    ca = atoms[atoms.atom_name == "CA"]
    out: list[tuple[str, str]] = []
    for chain_id in dict.fromkeys(ca.chain_id):  # unique chains, ordered
        chain_ca = ca[ca.chain_id == chain_id]
        seq = "".join(_3TO1.get(name, "X") for name in chain_ca.res_name)
        out.append((str(chain_id), seq))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    pdb_dir = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(pdb_dir, "sequences.csv")

    pdb_paths = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_paths:
        print(f"no *.pdb files in {pdb_dir}", file=sys.stderr)
        sys.exit(1)

    rows = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb", "chain", "length", "sequence"])
        for path in pdb_paths:
            for chain_id, seq in chain_sequences(path):
                w.writerow([os.path.basename(path), chain_id, len(seq), seq])
                rows += 1
    print(f"wrote {rows} rows ({len(pdb_paths)} pdbs) -> {out_csv}")


if __name__ == "__main__":
    main()
