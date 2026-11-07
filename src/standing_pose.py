"""Coordinated crouch targets and closed-linkage reset angles for this robot."""

import cmath
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def crouch_joint_positions():
    """Solve the passive four-bar joints instead of resetting them to zero."""
    source = Path(__file__).resolve().parents[1] / "robots/humanoid_legs_5/humanoid_legs_5.xml"
    tree = ET.parse(source)
    positions = {}

    def yz(element):
        _, y, z = map(float, element.attrib["pos"].split())
        return complex(y, z)

    for side in ("left", "right"):
        ankle_angle = 0.2
        foot = tree.find(f".//body[@name='{side}_foot']")
        heel = foot.find(f"body[@name='{side}_heel_linkage']")
        crank = tree.find(f".//body[@name='{side}_ankle_crank']")
        pivot = yz(foot.find("joint"))
        lower_pin = pivot + (yz(heel) - pivot) * cmath.exp(1j * ankle_angle)
        crank_pivot = yz(crank)
        bar_vector = yz(heel.find("site"))
        crank_vector = yz(crank.find("site"))
        distance = abs(crank_pivot - lower_pin)
        direction = (crank_pivot - lower_pin) / distance
        along = (abs(bar_vector)**2 - abs(crank_vector)**2 + distance**2) / (2 * distance)
        height_sq = abs(bar_vector)**2 - along**2
        if height_sq < 0:
            raise ValueError("Crouch ankle target cannot close its linkage")
        center = lower_pin + along * direction
        choices = [center + sign * 1j * direction * math.sqrt(height_sq) for sign in (-1, 1)]
        expected = crank_pivot + crank_vector * cmath.exp(1j * ankle_angle)
        upper_pin = min(choices, key=lambda point: abs(point - expected))
        positions.update({
            f"{side}_hip1_joint": 0.2,
            f"{side}_knee_joint": -0.4,
            f"{side}_ankle_joint": ankle_angle,
            f"{side}_ankle_crank_joint": cmath.phase((upper_pin - crank_pivot) / crank_vector),
            f"{side}_heel_bar_joint": cmath.phase((upper_pin - lower_pin) / bar_vector) - ankle_angle,
        })
    return positions
