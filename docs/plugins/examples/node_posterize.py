"""Example Python node: posterize stored spherical-harmonic DC colour."""

import lichtfeld as lf


class PosterizeExample(lf.nodes.Node):
    id = "example.posterize"
    label = "Posterize Example"
    category = "Colour"

    inputs = [
        lf.nodes.Input("Geometry", "geometry"),
        lf.nodes.Input("Levels", "int", 4, min=2, max=32),
    ]
    outputs = [lf.nodes.Output("Geometry", "geometry")]

    def execute(self, ctx):
        geometry = ctx.input("Geometry")
        if geometry.splats is None:
            return {"Geometry": geometry}
        levels = max(2, int(ctx.input("Levels")))
        colour = (geometry.splats.sh0 * levels).floor() / levels
        return {
            "Geometry": geometry.replace(
                splats=geometry.splats.replace(sh0=colour)
            )
        }


def register():
    lf.nodes.register_node(PosterizeExample)


def unregister():
    lf.nodes.unregister_node(PosterizeExample.id)
