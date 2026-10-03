"""Built-in Python node used to exercise the safe callback path."""

import lichtfeld as lf


class Posterize:
    id = "lfs.posterize"
    label = lf.ui.tr("nodes.posterize.label")
    category = "Colour"
    description = lf.ui.tr("nodes.posterize.description")
    help = lf.ui.tr("nodes.posterize.help")

    inputs = [
        lf.nodes.Input("Geometry", "geometry", description=lf.ui.tr("nodes.posterize.inputs.Geometry")),
        lf.nodes.Input("Selection", "float", 1.0, min=0.0, max=1.0, field=True,
                       description=lf.ui.tr("nodes.posterize.inputs.Selection")),
        lf.nodes.Input("Levels", "int", 4, min=2, max=32,
                       description=lf.ui.tr("nodes.posterize.inputs.Levels")),
    ]
    outputs = [lf.nodes.Output("Geometry", "geometry", description=lf.ui.tr("nodes.posterize.outputs.Geometry"))]

    def execute(self, ctx):
        geometry = ctx.input("Geometry")
        if geometry is None or geometry.splats is None:
            return {"Geometry": geometry}
        levels = max(2, int(ctx.input("Levels")))
        splats = geometry.splats
        weight = ctx.field("Selection", splats).clamp(0.0, 1.0)
        colour = (splats.sh0 * 0.28209479177387814 + 0.5).clamp(0.0, 1.0)
        colour = (colour * float(levels - 1)).round() / float(levels - 1)
        quantized = (colour - 0.5) / 0.28209479177387814
        sh0 = splats.sh0 + (quantized - splats.sh0) * weight.unsqueeze(1)
        shn = splats.shN * (1.0 - weight).reshape((-1, 1, 1))
        return {
            "Geometry": geometry.replace(
                splats=splats.replace(sh0=sh0, shN=shn),
            )
        }


lf.nodes.register_node(Posterize)
