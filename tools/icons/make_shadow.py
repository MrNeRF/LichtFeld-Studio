from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


SIZE = 192
SHAPE_SIZE = 96
CORNER_RADIUS = 8
BLUR_SIGMA = 12


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "src/visualizer/gui/assets/rmlui/soft_shadow.png"

    mask = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask)
    margin = (SIZE - SHAPE_SIZE) // 2
    draw.rounded_rectangle(
        (margin, margin, margin + SHAPE_SIZE - 1, margin + SHAPE_SIZE - 1),
        radius=CORNER_RADIUS,
        fill=255,
    )
    alpha = mask.filter(ImageFilter.GaussianBlur(BLUR_SIGMA))

    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow.putalpha(alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shadow.save(output_path)


if __name__ == "__main__":
    main()
