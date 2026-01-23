#!/usr/bin/env python

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


def _parse_range(s: str) -> tuple[float, float]:
    parts = [p for p in s.replace(",", " ").split() if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two floats for range, got: {s!r}")
    lo, hi = float(parts[0]), float(parts[1])
    return lo, hi


def _fmt(x: float) -> str:
    # Keep MJCF readable and stable.
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6g}"


def _fmt_range(lo: float, hi: float) -> str:
    return f"{_fmt(lo)} {_fmt(hi)}"


@dataclass
class PiperXmlSnapshot:
    piper_joint_frictionloss: float | None
    piper_joint_armature: float | None
    piper_forcerange: tuple[float, float] | None
    finger_forcerange: tuple[float, float] | None
    actuators: list[dict[str, object]]


def _find_default_class(root: ET.Element, class_name: str) -> ET.Element | None:
    # The file uses nested <default> tags. We just search for the one with matching class.
    for node in root.findall(".//default"):
        if node.get("class") == class_name:
            return node
    return None


def snapshot_piper_xml(path: Path) -> PiperXmlSnapshot:
    tree = ET.parse(path)
    root = tree.getroot()
    return snapshot_from_root(root)


def snapshot_from_root(root: ET.Element) -> PiperXmlSnapshot:
    piper = _find_default_class(root, "piper")
    finger = _find_default_class(root, "finger")

    piper_joint = piper.find("joint") if piper is not None else None
    piper_pos = piper.find("position") if piper is not None else None

    finger_pos = finger.find("position") if finger is not None else None

    piper_joint_frictionloss = float(piper_joint.get("frictionloss")) if piper_joint is not None and piper_joint.get("frictionloss") else None
    piper_joint_armature = float(piper_joint.get("armature")) if piper_joint is not None and piper_joint.get("armature") else None
    piper_forcerange = _parse_range(piper_pos.get("forcerange")) if piper_pos is not None and piper_pos.get("forcerange") else None
    finger_forcerange = _parse_range(finger_pos.get("forcerange")) if finger_pos is not None and finger_pos.get("forcerange") else None

    actuators: list[dict[str, object]] = []
    act_root = root.find("actuator")
    if act_root is not None:
        for pos in act_root.findall("position"):
            name = pos.get("name", "")
            joint = pos.get("joint", "")
            klass = pos.get("class", "")
            kp = float(pos.get("kp")) if pos.get("kp") else None
            kv = float(pos.get("kv")) if pos.get("kv") else None
            actuators.append({"name": name, "joint": joint, "class": klass, "kp": kp, "kv": kv})

    return PiperXmlSnapshot(
        piper_joint_frictionloss=piper_joint_frictionloss,
        piper_joint_armature=piper_joint_armature,
        piper_forcerange=piper_forcerange,
        finger_forcerange=finger_forcerange,
        actuators=actuators,
    )


def apply_scales(
    *,
    path_in: Path,
    path_out: Path,
    kp_scale: float,
    kv_scale: float,
    piper_forcerange_scale: float,
    finger_forcerange_scale: float,
    frictionloss_scale: float,
    armature_scale: float,
    write: bool,
    in_place: bool,
) -> None:
    tree = ET.parse(path_in)
    root = tree.getroot()

    def _scale_attr(node: ET.Element, attr: str, scale: float) -> bool:
        v = node.get(attr)
        if v is None:
            return False
        try:
            fv = float(v)
        except Exception:
            return False
        node.set(attr, _fmt(fv * scale))
        return True

    # Defaults
    piper = _find_default_class(root, "piper")
    finger = _find_default_class(root, "finger")

    if piper is not None:
        j = piper.find("joint")
        if j is not None:
            _scale_attr(j, "frictionloss", frictionloss_scale)
            _scale_attr(j, "armature", armature_scale)

        p = piper.find("position")
        if p is not None and p.get("forcerange"):
            lo, hi = _parse_range(p.get("forcerange"))
            p.set("forcerange", _fmt_range(lo * piper_forcerange_scale, hi * piper_forcerange_scale))

    if finger is not None:
        p = finger.find("position")
        if p is not None and p.get("forcerange"):
            lo, hi = _parse_range(p.get("forcerange"))
            p.set("forcerange", _fmt_range(lo * finger_forcerange_scale, hi * finger_forcerange_scale))

    # Actuators
    act_root = root.find("actuator")
    if act_root is not None:
        for pos in act_root.findall("position"):
            if pos.get("kp"):
                _scale_attr(pos, "kp", kp_scale)
            if pos.get("kv"):
                _scale_attr(pos, "kv", kv_scale)

    snap_before = snapshot_piper_xml(path_in)
    snap_after = snapshot_from_root(root)

    # Write (or dry run)
    if in_place:
        path_out = path_in

    if write:
        if in_place:
            bak = path_in.with_suffix(path_in.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path_in, bak)
        tree.write(path_out, encoding="utf-8", xml_declaration=False)
        snap_after = snapshot_piper_xml(path_out)

    print("=== piper.xml (before) ===")
    print_snapshot(snap_before)
    print("")
    print("=== piper.xml (after) ===")
    print_snapshot(snap_after)
    if write:
        print("")
        print(f"Wrote: {path_out}")
        if in_place:
            print(f"Backup: {path_in.with_suffix(path_in.suffix + '.bak')}")
        print("NOTE: LuckyEngine will need to reload the scene / restart to pick up MJCF changes.")
    else:
        print("")
        print("(dry-run; pass --write to produce output)")


def print_snapshot(s: PiperXmlSnapshot) -> None:
    print(f"- piper default joint frictionloss: {s.piper_joint_frictionloss}")
    print(f"- piper default joint armature:     {s.piper_joint_armature}")
    print(f"- piper default position forcerange:{s.piper_forcerange}")
    print(f"- finger default position forcerange:{s.finger_forcerange}")
    print("- actuators:")
    for a in s.actuators:
        print(f"  - {a.get('name')} ({a.get('class')}) joint={a.get('joint')} kp={a.get('kp')} kv={a.get('kv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect (and optionally patch) Piper MJCF parameters in piper.xml.")
    parser.add_argument(
        "--piper_xml",
        type=Path,
        default=Path(r"d:\rsl_rl\LuckyEngine\LuckyEditor\RobotSandbox\Assets\agilex_piper\piper.xml"),
        help="Path to piper.xml",
    )
    parser.add_argument("--out_xml", type=Path, default=None, help="Output xml path (default: alongside input).")
    parser.add_argument("--kp_scale", type=float, default=1.0, help="Scale all actuator kp by this factor.")
    parser.add_argument("--kv_scale", type=float, default=1.0, help="Scale all actuator kv by this factor.")
    parser.add_argument(
        "--piper_forcerange_scale",
        type=float,
        default=1.0,
        help='Scale default class="piper" position forcerange by this factor.',
    )
    parser.add_argument(
        "--finger_forcerange_scale",
        type=float,
        default=1.0,
        help='Scale default class="finger" position forcerange by this factor.',
    )
    parser.add_argument("--frictionloss_scale", type=float, default=1.0, help="Scale default piper joint frictionloss.")
    parser.add_argument("--armature_scale", type=float, default=1.0, help="Scale default piper joint armature.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write the modified XML (default is dry-run).",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Write changes in-place (creates a .bak once). If not set, writes to --out_xml (or auto-generated).",
    )
    args = parser.parse_args()

    p = Path(args.piper_xml)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    out = args.out_xml
    if out is None and not args.in_place:
        out = p.with_name(p.stem + f"_kp{args.kp_scale:g}_fr{args.piper_forcerange_scale:g}.xml")

    if out is None:
        out = p

    apply_scales(
        path_in=p,
        path_out=Path(out),
        kp_scale=float(args.kp_scale),
        kv_scale=float(args.kv_scale),
        piper_forcerange_scale=float(args.piper_forcerange_scale),
        finger_forcerange_scale=float(args.finger_forcerange_scale),
        frictionloss_scale=float(args.frictionloss_scale),
        armature_scale=float(args.armature_scale),
        write=bool(args.write),
        in_place=bool(args.in_place),
    )


if __name__ == "__main__":
    main()


