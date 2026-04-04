import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


EXPECTED_LABELS: Tuple[str, ...] = ("shaft", "wrist", "ee", "tip1", "tip2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a CVAT XML annotation file and report duplicate keypoint "
            "labels within the same frame instance."
        )
    )
    parser.add_argument(
        "xml_path",
        help="Path to the XML annotation file to validate.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(EXPECTED_LABELS),
        help=(
            "Point labels to validate. Defaults to: " f"{', '.join(EXPECTED_LABELS)}."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove duplicate points using the built-in heuristic.",
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Where to write the cleaned XML when using --fix. "
            "Defaults to <input_stem>_clean.xml next to the input file."
        ),
    )
    return parser.parse_args()


def parse_xy(points_str: str) -> Tuple[float, float]:
    x_str, y_str = points_str.split(",")
    return float(x_str), float(y_str)


def collect_duplicate_labels_from_root(
    root: ET.Element, labels_to_check: Iterable[str]
) -> List[Dict[str, object]]:
    valid_labels = set(labels_to_check)
    issues: List[Dict[str, object]] = []

    for image_el in root.findall("image"):
        image_name = image_el.attrib.get("name", "<unknown>")
        image_id = image_el.attrib.get("id", "<unknown>")

        counts_by_instance: Dict[str, Counter] = defaultdict(Counter)

        for point_el in image_el.findall("points"):
            label = point_el.attrib.get("label")
            instance = point_el.attrib.get("instance")

            if label not in valid_labels or instance is None:
                continue

            counts_by_instance[instance][label] += 1

        for instance, label_counts in sorted(
            counts_by_instance.items(), key=lambda item: item[0]
        ):
            duplicate_labels = {
                label: count
                for label, count in sorted(label_counts.items())
                if count > 1
            }
            if not duplicate_labels:
                continue

            issues.append(
                {
                    "image_id": image_id,
                    "image_name": image_name,
                    "instance": instance,
                    "duplicate_labels": duplicate_labels,
                }
            )

    return issues


def collect_duplicate_labels(
    xml_path: str, labels_to_check: Iterable[str]
) -> List[Dict[str, object]]:
    tree = ET.parse(xml_path)
    return collect_duplicate_labels_from_root(tree.getroot(), labels_to_check)


def format_issue(issue: Dict[str, object]) -> str:
    duplicates = ", ".join(
        f"{label} x{count}" for label, count in issue["duplicate_labels"].items()
    )
    return (
        f"image_id={issue['image_id']} image_name={issue['image_name']} "
        f"instance={issue['instance']} duplicate_labels=[{duplicates}]"
    )


def choose_best_duplicate(
    duplicate_points: Sequence[ET.Element],
    reference_points: Sequence[ET.Element],
    instance: str,
) -> ET.Element:
    if reference_points:
        scored_candidates = []
        reference_xy = [parse_xy(point.attrib["points"]) for point in reference_points]

        for point in duplicate_points:
            px, py = parse_xy(point.attrib["points"])
            score = sum(math.dist((px, py), ref_xy) for ref_xy in reference_xy)
            scored_candidates.append((score, px, point))

        reverse_x = instance == "2"
        scored_candidates.sort(
            key=lambda item: (item[0], -item[1] if reverse_x else item[1])
        )
        return scored_candidates[0][2]

    point_with_x = [
        (parse_xy(point.attrib["points"])[0], point) for point in duplicate_points
    ]
    if instance == "2":
        return max(point_with_x, key=lambda item: item[0])[1]
    return min(point_with_x, key=lambda item: item[0])[1]


def deduplicate_points(
    root: ET.Element, labels_to_check: Iterable[str]
) -> List[Dict[str, object]]:
    valid_labels = set(labels_to_check)
    removals: List[Dict[str, object]] = []

    for image_el in root.findall("image"):
        image_name = image_el.attrib.get("name", "<unknown>")
        image_id = image_el.attrib.get("id", "<unknown>")

        points_by_instance_and_label: Dict[str, Dict[str, List[ET.Element]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        for point_el in image_el.findall("points"):
            label = point_el.attrib.get("label")
            instance = point_el.attrib.get("instance")
            if label not in valid_labels or instance is None:
                continue
            points_by_instance_and_label[instance][label].append(point_el)

        for instance, points_by_label in sorted(
            points_by_instance_and_label.items(), key=lambda item: item[0]
        ):
            instance_reference_points = [
                point
                for label, points in points_by_label.items()
                if len(points) == 1
                for point in points
            ]

            for label, duplicate_points in sorted(points_by_label.items()):
                if len(duplicate_points) <= 1:
                    continue

                reference_points = instance_reference_points + [
                    point
                    for other_label, points in points_by_label.items()
                    if other_label != label and len(points) > 1
                    for point in points
                ]
                point_to_keep = choose_best_duplicate(
                    duplicate_points, reference_points, instance
                )

                removed_points = []
                for point_el in duplicate_points:
                    if point_el is point_to_keep:
                        continue
                    removed_points.append(point_el.attrib["points"])
                    image_el.remove(point_el)

                removals.append(
                    {
                        "image_id": image_id,
                        "image_name": image_name,
                        "instance": instance,
                        "label": label,
                        "kept": point_to_keep.attrib["points"],
                        "removed": removed_points,
                    }
                )

    return removals


def format_removal(removal: Dict[str, object]) -> str:
    removed = ", ".join(removal["removed"])
    return (
        f"image_id={removal['image_id']} image_name={removal['image_name']} "
        f"instance={removal['instance']} label={removal['label']} "
        f"kept={removal['kept']} removed=[{removed}]"
    )


def default_output_path(xml_path: str) -> str:
    stem, ext = os.path.splitext(xml_path)
    return f"{stem}_clean{ext}"


def main() -> int:
    args = parse_args()
    tree = ET.parse(args.xml_path)
    root = tree.getroot()
    issues = collect_duplicate_labels_from_root(root, args.labels)

    if not args.fix:
        if not issues:
            print(
                "Validation passed: no duplicate point labels were found within the "
                "same image instance."
            )
            return 0

        print(
            f"Validation failed: found {len(issues)} image-instance pair(s) with "
            "duplicate point labels."
        )
        for issue in issues:
            print(format_issue(issue))

        return 1

    removals = deduplicate_points(root, args.labels)
    output_path = args.output_path or default_output_path(args.xml_path)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    remaining_issues = collect_duplicate_labels_from_root(root, args.labels)

    if not removals:
        print(
            "No duplicate point labels were found, so no cleaned file changes were "
            f"needed. Wrote XML to {output_path}."
        )
        return 0

    print(
        f"Removed duplicate point labels from {len(removals)} image-instance-label "
        f"group(s). Wrote cleaned XML to {output_path}."
    )
    for removal in removals:
        print(format_removal(removal))

    if remaining_issues:
        print(
            f"Warning: {len(remaining_issues)} duplicate image-instance pair(s) still "
            "remain after cleanup."
        )
        for issue in remaining_issues:
            print(format_issue(issue))
        return 1

    print("Post-fix validation passed: no duplicate point labels remain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
