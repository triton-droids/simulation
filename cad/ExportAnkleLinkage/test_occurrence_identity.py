"""Offline regression for linked CAD documents with colliding entity tokens."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


tree = ast.parse(Path(__file__).with_name("ExportAnkleLinkage.py").read_text())
helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in ("occurrence_path", "collect_occurrences")], type_ignores=[])
namespace = {}
exec(compile(helpers, "exporter_helpers", "exec"), namespace)


def occurrence(name, children=()):
    matrix = SimpleNamespace(asArray=lambda: [1, 0, 0, 0, 0, 1, 0, 0,
                                             0, 0, 1, 0, 0, 0, 0, 1])
    return SimpleNamespace(component=SimpleNamespace(name=name, entityToken="same-token"),
                           fullPathName=name + ":1", assemblyContext=None,
                           transform2=matrix, nativeObject=None, childOccurrences=children)


class OccurrenceIdentityTest(unittest.TestCase):
    def test_distinct_linked_components_with_same_token_survive(self):
        root = occurrence("Renamed assembly", [occurrence("R Foot Sole"),
                          occurrence("R Heel Linkage"), occurrence("L Heel Linkage")])
        records, components = namespace["collect_occurrences"](root)
        self.assertEqual(len(components), 4)
        self.assertEqual(len({r["component_id"] for r in records}), 4)
        self.assertEqual([c.name for _, c, _ in components],
                         ["Renamed assembly", "R Foot Sole", "R Heel Linkage", "L Heel Linkage"])

    def test_repeated_instance_keeps_its_own_placement(self):
        first, second = occurrence("Bearing"), occurrence("Bearing")
        second.component = first.component
        second.fullPathName = "Bearing:2"
        records, _ = namespace["collect_occurrences"](occurrence("Root", [first, second]))
        self.assertNotEqual(records[1]["component_id"], records[2]["component_id"])
        self.assertEqual(records[2]["path"], "Bearing:2")


if __name__ == "__main__":
    unittest.main()
