import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


EXPECTED_LABELS: Tuple[str, ...] = ("shaft", "wrist", "ee", "tip1", "tip2")
EXPECTED_INSTANCES: Tuple[str, ...] = ("1", "2")


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
    parser.add_argument(
        "--fill-mode",
        choices=("interpolate", "last"),
        default="interpolate",
        help=(
            "How to fill missing points during --fix. "
            "'interpolate' uses linear interpolation between previous and next "
            "observations. 'last' copies the last observed location forward."
        ),
    )
    return parser.parse_args()


def parse_xy(points_str: str) -> Tuple[float, float]:
    x_str, y_str = points_str.split(",")
    return float(x_str), float(y_str)


def format_xy(x: float, y: float) -> str:
    return f"{x:.2f},{y:.2f}"


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


def collect_missing_labels_from_root(
    root: ET.Element,
    labels_to_check: Iterable[str],
    instances_to_check: Iterable[str] = EXPECTED_INSTANCES,
) -> List[Dict[str, object]]:
    valid_labels = tuple(labels_to_check)
    expected_label_set = set(valid_labels)
    target_instances = tuple(instances_to_check)
    issues: List[Dict[str, object]] = []

    for image_el in root.findall("image"):
        image_name = image_el.attrib.get("name", "<unknown>")
        image_id = image_el.attrib.get("id", "<unknown>")

        labels_by_instance: Dict[str, set] = {
            instance: set() for instance in target_instances
        }

        for point_el in image_el.findall("points"):
            label = point_el.attrib.get("label")
            instance = point_el.attrib.get("instance")

            if label not in expected_label_set or instance not in labels_by_instance:
                continue

            labels_by_instance[instance].add(label)

        for instance in target_instances:
            missing_labels = [
                label
                for label in valid_labels
                if label not in labels_by_instance[instance]
            ]
            if not missing_labels:
                continue

            issues.append(
                {
                    "image_id": image_id,
                    "image_name": image_name,
                    "instance": instance,
                    "missing_labels": missing_labels,
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


def format_missing_issue(issue: Dict[str, object]) -> str:
    missing = ", ".join(issue["missing_labels"])
    return (
        f"image_id={issue['image_id']} image_name={issue['image_name']} "
        f"instance={issue['instance']} missing_labels=[{missing}]"
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


def build_filled_point(
    label: str,
    instance: str,
    point_str: str,
    prev_point_el: ET.Element,
    next_point_el: ET.Element | None = None,
) -> ET.Element:
    point_el = ET.Element("points")
    point_el.attrib["label"] = label
    point_el.attrib["source"] = prev_point_el.attrib.get(
        "source",
        next_point_el.attrib.get("source", "auto") if next_point_el is not None else "auto",
    )
    point_el.attrib["occluded"] = "1"
    point_el.attrib["points"] = point_str
    point_el.attrib["z_order"] = prev_point_el.attrib.get(
        "z_order",
        next_point_el.attrib.get("z_order", "0") if next_point_el is not None else "0",
    )
    point_el.attrib["instance"] = instance
    return point_el


def fill_missing_points(
    root: ET.Element,
    labels_to_check: Iterable[str],
    fill_mode: str,
    instances_to_check: Iterable[str] = EXPECTED_INSTANCES,
) -> List[Dict[str, object]]:
    valid_labels = tuple(labels_to_check)
    target_instances = tuple(instances_to_check)
    image_elements = root.findall("image")
    insertions: List[Dict[str, object]] = []

    existing_points: Dict[Tuple[str, str], Dict[int, ET.Element]] = {
        (instance, label): {}
        for instance in target_instances
        for label in valid_labels
    }

    for frame_idx, image_el in enumerate(image_elements):
        for point_el in image_el.findall("points"):
            label = point_el.attrib.get("label")
            instance = point_el.attrib.get("instance")
            key = (instance, label)
            if key not in existing_points:
                continue
            existing_points[key][frame_idx] = point_el

    for instance in target_instances:
        for label in valid_labels:
            frame_to_point = existing_points[(instance, label)]
            present_frames = sorted(frame_to_point)

            if fill_mode == "interpolate":
                if len(present_frames) < 2:
                    continue

                for prev_frame_idx, next_frame_idx in zip(
                    present_frames, present_frames[1:]
                ):
                    frame_gap = next_frame_idx - prev_frame_idx
                    if frame_gap <= 1:
                        continue

                    prev_point_el = frame_to_point[prev_frame_idx]
                    next_point_el = frame_to_point[next_frame_idx]
                    prev_x, prev_y = parse_xy(prev_point_el.attrib["points"])
                    next_x, next_y = parse_xy(next_point_el.attrib["points"])

                    for missing_frame_idx in range(prev_frame_idx + 1, next_frame_idx):
                        alpha = (missing_frame_idx - prev_frame_idx) / frame_gap
                        filled_x = prev_x + alpha * (next_x - prev_x)
                        filled_y = prev_y + alpha * (next_y - prev_y)
                        point_str = format_xy(filled_x, filled_y)
                        image_el = image_elements[missing_frame_idx]

                        inserted_point = build_filled_point(
                            label=label,
                            instance=instance,
                            point_str=point_str,
                            prev_point_el=prev_point_el,
                            next_point_el=next_point_el,
                        )
                        image_el.append(inserted_point)

                        insertions.append(
                            {
                                "image_id": image_el.attrib.get("id", "<unknown>"),
                                "image_name": image_el.attrib.get("name", "<unknown>"),
                                "instance": instance,
                                "label": label,
                                "points": point_str,
                                "fill_mode": fill_mode,
                                "prev_image_name": image_elements[prev_frame_idx].attrib.get(
                                    "name", "<unknown>"
                                ),
                                "next_image_name": image_elements[next_frame_idx].attrib.get(
                                    "name", "<unknown>"
                                ),
                            }
                        )
                continue

            if fill_mode == "last":
                if not present_frames:
                    continue

                next_present_idx = 0
                last_seen_frame_idx = None
                last_seen_point_el = None

                for frame_idx, image_el in enumerate(image_elements):
                    if (
                        next_present_idx < len(present_frames)
                        and frame_idx == present_frames[next_present_idx]
                    ):
                        last_seen_frame_idx = frame_idx
                        last_seen_point_el = frame_to_point[frame_idx]
                        next_present_idx += 1
                        continue

                    if last_seen_point_el is None:
                        continue

                    prev_x, prev_y = parse_xy(last_seen_point_el.attrib["points"])
                    point_str = format_xy(prev_x, prev_y)

                    inserted_point = build_filled_point(
                        label=label,
                        instance=instance,
                        point_str=point_str,
                        prev_point_el=last_seen_point_el,
                    )
                    image_el.append(inserted_point)

                    insertions.append(
                        {
                            "image_id": image_el.attrib.get("id", "<unknown>"),
                            "image_name": image_el.attrib.get("name", "<unknown>"),
                            "instance": instance,
                            "label": label,
                            "points": point_str,
                            "fill_mode": fill_mode,
                            "prev_image_name": image_elements[last_seen_frame_idx].attrib.get(
                                "name", "<unknown>"
                            ),
                        }
                    )

                continue

            raise ValueError(f"Unsupported fill mode: {fill_mode}")

    return insertions


def format_removal(removal: Dict[str, object]) -> str:
    removed = ", ".join(removal["removed"])
    return (
        f"image_id={removal['image_id']} image_name={removal['image_name']} "
        f"instance={removal['instance']} label={removal['label']} "
        f"kept={removal['kept']} removed=[{removed}]"
    )


def format_insertion(insertion: Dict[str, object]) -> str:
    if insertion["fill_mode"] == "last":
        return (
            f"image_id={insertion['image_id']} image_name={insertion['image_name']} "
            f"instance={insertion['instance']} label={insertion['label']} "
            f"points={insertion['points']} "
            f"copied_from={insertion['prev_image_name']}"
        )

    return (
        f"image_id={insertion['image_id']} image_name={insertion['image_name']} "
        f"instance={insertion['instance']} label={insertion['label']} "
        f"points={insertion['points']} "
        f"interpolated_between=[{insertion['prev_image_name']}, {insertion['next_image_name']}]"
    )


def default_output_path(xml_path: str) -> str:
    stem, ext = os.path.splitext(xml_path)
    return f"{stem}_clean{ext}"


def main() -> int:
    args = parse_args()
    tree = ET.parse(args.xml_path)
    root = tree.getroot()
    duplicate_issues = collect_duplicate_labels_from_root(root, args.labels)
    missing_issues = collect_missing_labels_from_root(root, args.labels)

    if not args.fix:
        if not duplicate_issues and not missing_issues:
            print(
                "Validation passed: no duplicate or missing point labels were found "
                "for instances 1 and 2."
            )
            return 0

        total_issues = len(duplicate_issues) + len(missing_issues)
        print(f"Validation failed: found {total_issues} image-instance issue(s).")
        if duplicate_issues:
            print(
                f"Duplicate label issues: {len(duplicate_issues)} image-instance pair(s)."
            )
            for issue in duplicate_issues:
                print(format_issue(issue))
        if missing_issues:
            print(
                f"Missing label issues: {len(missing_issues)} image-instance pair(s)."
            )
            for issue in missing_issues:
                print(format_missing_issue(issue))

        return 1

    removals = deduplicate_points(root, args.labels)
    insertions = fill_missing_points(root, args.labels, args.fill_mode)
    output_path = args.output_path or default_output_path(args.xml_path)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    remaining_duplicate_issues = collect_duplicate_labels_from_root(root, args.labels)
    remaining_missing_issues = collect_missing_labels_from_root(root, args.labels)

    if not removals and not insertions and not remaining_missing_issues:
        print(
            "No duplicate point labels or fillable missing labels were found, "
            f"so no cleaned file changes were needed. Wrote XML to {output_path}."
        )
        return 0

    print(
        f"Wrote cleaned XML to {output_path}. Removed duplicates from "
        f"{len(removals)} image-instance-label group(s) and filled "
        f"{len(insertions)} missing point(s) using '{args.fill_mode}'."
    )
    for removal in removals:
        print(format_removal(removal))
    for insertion in insertions:
        print(format_insertion(insertion))

    if remaining_duplicate_issues or remaining_missing_issues:
        print(
            f"Post-fix validation found "
            f"{len(remaining_duplicate_issues) + len(remaining_missing_issues)} "
            "remaining image-instance issue(s)."
        )
        if remaining_duplicate_issues:
            print(
                f"Remaining duplicate label issues: "
                f"{len(remaining_duplicate_issues)} image-instance pair(s)."
            )
            for issue in remaining_duplicate_issues:
                print(format_issue(issue))
        if remaining_missing_issues:
            print(
                f"Remaining missing label issues: "
                f"{len(remaining_missing_issues)} image-instance pair(s)."
            )
            for issue in remaining_missing_issues:
                print(format_missing_issue(issue))
        return 1

    print(
        "Post-fix validation passed: no duplicate or missing point labels remain "
        "for instances 1 and 2."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
