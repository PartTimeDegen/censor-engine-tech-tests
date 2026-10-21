import hashlib
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted


def read_images(
    sub_folder: str | None = "base",
    limit: int | None = None,
):
    path = Path("src/data/inputs")
    if sub_folder is not None:
        path = path / sub_folder

    images = natsorted(
        [image for image in path.glob("*") if str(image) != ".gitkeep"]
    )
    if limit is None:
        return images

    return images[:limit]


def write_image(input_path: Path, image, folder: str):
    path_folder = Path(f"src/data/outputs/{folder}")
    path_folder.mkdir(exist_ok=True, parents=True)
    final_folder = f"{path_folder}/{input_path.parts[-1]}"
    cv2.imwrite(final_folder, image)
    print(f"Saved File: {final_folder}")


def get_region_color(region_name: str) -> tuple[int, int, int]:
    """Generate a deterministic BGR color from a region name."""
    digest = hashlib.sha256(region_name.encode("utf-8")).digest()

    # OpenCV uses BGR; keep colors reasonably bright.
    return tuple(64 + (byte % 192) for byte in digest[:3])


def draw_oriented_box(image, points, color, label=None, draw_labels=True):
    """Draw a minimum-area rotated rectangle around points."""
    if len(points) < 3:
        return

    points = np.asarray(points, dtype=np.float32)

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.int32)

    cv2.polylines(
        image,
        [box],
        isClosed=True,
        color=color,
        thickness=2,
    )

    if label and draw_labels:
        # Put the label above the uppermost corner.
        top_index = np.argmin(box[:, 1])
        label_x, label_y = box[top_index]

        cv2.putText(
            image,
            label,
            (int(label_x), max(15, int(label_y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_fitted_ellipse(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = -1,
) -> np.ndarray:
    """Draw a fitted ellipse around a set of points."""
    output = image.copy()
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(points) < 5:
        return output

    ellipse = cv2.fitEllipse(points.astype(np.float32))
    cv2.ellipse(output, ellipse, color, thickness, cv2.LINE_AA)

    return output


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
    closed: bool = True,
) -> np.ndarray:
    """Draw a polyline around the outermost points using a convex hull."""
    output = image.copy()
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(points) < 3:
        return output

    points = np.round(points).astype(np.int32)

    # Keep only the outermost points, ordered around the boundary.
    hull = cv2.convexHull(points)

    # cv2.polylines(
    #     output,
    #     [hull],
    #     isClosed=closed,
    #     color=color,
    #     thickness=thickness,
    #     lineType=cv2.LINE_AA,
    # )
    cv2.fillConvexPoly(
        output,
        hull,
        color=color,
        lineType=cv2.LINE_AA,
    )

    return output


if __name__ == "__main__":
    image = read_images()[0]
    im = cv2.imread(image)
    write_image(image, im, "thing")
