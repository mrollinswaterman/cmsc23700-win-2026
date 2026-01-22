from typing import Optional, Tuple, List
import numpy as np
from utils import *
from shapes import Shape, SVG, Triangle, Line, Circle


# NOTE feel free to write your own helper functions as long as they're in raster.py


def rasterize(
    svg_file: str,
    im_w: int,
    im_h: int,
    output_file: Optional[str] = None,
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    antialias: bool = True,
) -> np.ndarray:
    """
    :param svg_file: filename
    :param im_w: width of image to be rasterized
    :param im_h: height of image to be rasterized
    :param output_file: optional path to save numpy array
    :param background: background color, defaults to white (1, 1 ,1)
    :param antialias: whether to apply antialiasing, defaults to True
    :return: a numpy array of dimension (H,W,3) with RGB values in [0.0,1.0]
    """

    background_arr = np.array(background)
    shapes: List[Shape] = read_svg(svg_file)
    img = np.zeros((im_h, im_w, 3))
    img[:, :, :] = background_arr
    global svg
    svg = shapes[0]
    assert isinstance(svg, SVG)
    # the first shape in shapes is always the SVG object with the viewbox sizes

    # TODO: put your code here

    global image_size
    image_size = (im_w, im_h)

    for shape in shapes[1:]:
        # get shape bounding box in SVG space
        box: List[Point, Point] = get_bounding_box(shape, svg)
        # convert to img space to iterate over real pixels
        box = [
            to_img(box[0]),
            to_img(box[1]),
        ]

        for i in range(box[0].x, box[1].x + 1):
            for j in range(box[0].y, box[1].y + 1):
                pixel = Point((i, j))
                center = Point((i + 0.5, j + 0.5))

                if antialias:
                    # if aa, find the 3x3 grid of samples for the pixel
                    samples = get_samples(pixel)
                    count = 0.0

                    # check them all to find coverage
                    for pt in samples:
                        if point_in_shape(pt, shape, img):
                            count += 1.0
                    if count > 0:
                        coverage = count / 9.0
                        # find what the color at this pixel should be based on the coverage,
                        # and what's already there (equation from the assignment specs)
                        try:
                            color_at_pixel = (
                                (1 - coverage) * img[pixel.y, pixel.x]
                            ) + shape.color * coverage
                        except IndexError:
                            pass
                        fill_pixel(pixel, img, color_at_pixel)
                    continue

                # if no aa, simply check if the pixel's center is in the shape, then fill
                # that pixel if it is

                if point_in_shape(
                    to_svg(center),
                    shape,
                    img,
                ):
                    fill_pixel(pixel, img, shape)

    if output_file:
        save_image(output_file, img)

    return img


class Point:
    def __init__(self, coordinates: List[int | float] | Tuple[int | float]):
        self.x = coordinates[0]
        self.y = coordinates[1]

    def __str__(self):
        return f"({self.x}, {self.y})"


def fill_pixel(pixel: Point, img: np.array, color: Shape | np.ndarray):
    try:
        # if "color" is a shape object, use its color, otherwise treat "color" as an array
        match color:
            case Shape():
                img[pixel.y, pixel.x] = color.color
            case _:
                img[pixel.y, pixel.x] = color
    except IndexError as e:
        return False


def to_svg(pixel: Point) -> Point:
    """
    Docstring for to_svg

    :param pixel: An image space pixel
    :param svg: SVG object
    :param img_dimensions: Width and height of the image
    :return: an svg viewbox point
    :rtype: Point class instance
    """
    x_ratio = svg.w / image_size[0]
    y_ratio = svg.h / image_size[1]

    return Point((pixel.x * x_ratio, pixel.y * y_ratio))


def to_img(pt: Point) -> Point:
    # find the ratios of the two spaces
    x_ratio = svg.w / image_size[0]
    y_ratio = svg.h / image_size[1]

    # apply them to the given point to transform coordinate spaces
    return Point((int(np.ceil(pt.x / x_ratio)), int(np.ceil(pt.y / y_ratio))))


def get_bounding_box(shape: Shape, svg: SVG):

    match shape:
        case Triangle():
            # maybe could rewrite this to utilize the properties of numpy arrays
            max_x = max_y = 0
            min_x = 10000 if not svg else svg.w
            min_y = 10000 if not svg else svg.h

            for x, y in shape.pts:
                # check for new maxs and mins
                max_x = x if x > max_x else max_x
                min_x = x if x < min_x else min_x
                max_y = y if y > max_y else max_y
                min_y = y if y < min_y else min_y

        case Line():
            v1, v2, v3, v4 = get_rect_from_line(shape)
            min_x = min(v1.x, v2.x, v3.x, v4.x)
            max_x = max(v1.x, v2.x, v3.x, v4.x)
            min_y = min(v1.y, v2.y, v3.y, v4.y)
            max_y = max(v1.y, v2.y, v3.y, v4.y)

        case Circle():
            max_x = shape.center[0] + shape.radius + 4  # +4 to account for rounding
            min_x = shape.center[0] - shape.radius - 4  # see above
            max_y = shape.center[1] + shape.radius + 4  # ditto
            min_y = shape.center[1] - shape.radius - 4  # ditto

    return [
        Point((np.floor(min_x), np.floor(min_y))),
        Point((np.ceil(max_x), np.ceil(max_y))),
    ]


def point_in_shape(pt: Point, shape: Shape, verbose=False):

    match shape:
        case Triangle():
            # methodology from https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Equations_in_barycentric_coordinates

            v1 = Point(shape.pts[0])
            v2 = Point(shape.pts[1])
            v3 = Point(shape.pts[2])
            center = Point((pt.x + 0.5, pt.y + 0.5))

            bigA = area(v1, v2, v3)

            alpha = area(center, v2, v3) / bigA
            beta = area(v1, center, v3) / bigA
            gamma = area(v1, v2, center) / bigA

            return (
                (0 - 1e-15) <= alpha <= (1 + 1e-15)
                and (0 - 1e-15) <= beta <= (1 + 1e-15)
                and (0 - 1e-15) <= gamma <= (1 + 1e-15)
            )

        case Line():
            v1, v2, v3, v4 = get_rect_from_line(shape)

            triangle1_pts = [[v1.x, v1.y], [v3.x, v3.y], [v2.x, v2.y]]
            triangle1_pts = np.array(triangle1_pts, dtype=np.float64)
            triangle1 = Triangle(triangle1_pts, shape.color)

            triangle2_pts = [[v1.x, v1.y], [v4.x, v4.y], [v3.x, v3.y]]
            triangle2_pts = np.array(triangle2_pts, dtype=np.float64)
            triangle2 = Triangle(triangle2_pts, shape.color)

            return point_in_shape(pt, triangle2) or point_in_shape(pt, triangle1)

        case Circle():
            # c = √(xA − xB)^2 + (yA − yB)^2
            # so if the distance between the center of the point and the center of the circle
            # is less than the radius, point is in the circle!
            # methodolgy from here: https://stackoverflow.com/questions/77398101/write-code-to-test-if-a-point-is-in-circles
            return ((pt.x + 0.5) - shape.center[0]) ** 2 + (
                (pt.y + 0.5) - shape.center[1]
            ) ** 2 <= shape.radius**2


def area(v1: Point, v2: Point, v3: Point):
    # Returns the area of the trinagle created by the three parameter points
    return 0.5 * (
        v1.x * v2.y
        + v2.x * v3.y
        + v3.x * v1.y
        - v1.x * v3.y
        - v2.x * v1.y
        - v3.x * v2.y
    )


def get_rect_from_line(shape: Line) -> List[Point]:
    # Methodology for finding a rectangle from a line:
    # https://stackoverflow.com/questions/62398864/how-to-draw-a-thick-line-using-a-rectangle?rq=3

    # find the other points of the rectangle
    dx = shape.pts[0][0] - shape.pts[1][0]  # dx = x0 - x1
    dy = shape.pts[0][1] - shape.pts[1][1]  # dy = y0 - y1
    line_len = np.sqrt(
        dx * dx + dy * dy
    )  # length of the line based on the distance between endpoints

    py = -(shape.width // 2) * (dx) / (line_len)  # --> y-shift
    px = (shape.width // 2) * (dy) / (line_len)  # --> x-shift

    # apply the x and y shift to find the vertices of the rectangle

    v1 = Point((shape.pts[1][0] - px, shape.pts[1][1] - py))
    v2 = Point((shape.pts[1][0] + px, shape.pts[1][1] + py))
    v3 = Point((shape.pts[0][0] + px, shape.pts[0][1] + py))
    v4 = Point((shape.pts[0][0] - px, shape.pts[0][1] - py))

    return [v1, v2, v3, v4]


def get_samples(pixel: Point) -> List[Point]:
    samples = []
    for i in range(0, 3):
        column_x = pixel.x + (i * 0.5)  # x coordinate of the current column
        for j in range(0, 3):
            column_y = pixel.y + (j * 0.5)  # y coordinate of the current column
            samples.append(to_svg(Point((column_x, column_y))))

    return samples


if __name__ == "__main__":
    rasterize(
        "tests/test6.svg", 128, 128, output_file="your_output.png", antialias=True
    )
