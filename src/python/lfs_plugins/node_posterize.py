"""Built-in Python node used to exercise the safe callback path."""

import lichtfeld as lf


class Posterize:
    id = "lfs.posterize"
    label = lf.ui.tr("nodes.posterize.label")
    category = "Colour"
    description = lf.ui.tr("nodes.posterize.description")

    inputs = [
        lf.nodes.Input("Geometry", "geometry"),
        lf.nodes.Input("Levels", "int", 4, min=2, max=32),
    ]
    outputs = [lf.nodes.Output("Geometry", "geometry")]

    def execute(self, ctx):
        geometry = ctx.input("Geometry")
        if geometry is None or geometry.splats is None:
            return {"Geometry": geometry}
        levels = max(2, int(ctx.input("Levels")))
        sh0 = (geometry.splats.sh0 * float(levels)).floor() / float(levels)
        return {
            "Geometry": geometry.replace(
                splats=geometry.splats.replace(sh0=sh0),
            )
        }


lf.nodes.register_node(Posterize)
Posterize.inputs = None
Posterize.outputs = None
