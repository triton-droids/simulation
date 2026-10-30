"""Create a self-contained humanoid_legs_5 MJCF bundle for remote import."""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cad/ankle_linkage_inspection"
DEST = ROOT / "robots/humanoid_legs_5"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    mesh_dir = DEST / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    source_xml = SOURCE / "humanoid_legs_5.xml"
    tree = ET.parse(source_xml)
    root = tree.getroot()
    root.find("compiler").set("meshdir", "meshes")
    for mesh in root.find("asset").findall("mesh"):
        source_mesh = (source_xml.parent / mesh.get("file")).resolve(strict=True)
        target_name = mesh.get("name") + ".stl"
        shutil.copy2(source_mesh, mesh_dir / target_name)
        mesh.set("file", target_name)
    ET.indent(root)
    tree.write(DEST / "humanoid_legs_5.xml", encoding="unicode")
    for name in (
        "humanoid_legs_5_validation.json",
        "humanoid_legs_5.png",
        "humanoid_legs_5_ankle_detail.png",
        "humanoid_legs_5_ankle_sweep.mp4",
        "humanoid_legs_5_full_sweep.mp4",
    ):
        shutil.copy2(SOURCE / name, DEST / name)
    print(f"Packaged {len(root.find('asset').findall('mesh'))} meshes in {DEST}")


if __name__ == "__main__":
    main()
