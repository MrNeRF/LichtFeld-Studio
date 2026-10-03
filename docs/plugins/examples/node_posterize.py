"""Example Python node: posterize selected base RGB colours and flatten their SH."""

import lichtfeld as lf


class PosterizeExample(lf.nodes.Node):
    id = "example.posterize"
    label = "Posterize Example"
    category = "Colour"

    inputs = [
        lf.nodes.Input("Geometry", "geometry"),
        lf.nodes.Input("Selection", "float", 1.0, min=0.0, max=1.0, field=True),
        lf.nodes.Input("Levels", "int", 4, min=2, max=32),
    ]
    outputs = [lf.nodes.Output("Geometry", "geometry")]

    def execute(self, ctx):
        geometry = ctx.input("Geometry")
        if geometry.splats is None:
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
                splats=splats.replace(sh0=sh0, shN=shn)
            )
        }


def register():
    lf.nodes.register_node(PosterizeExample)


def unregister():
    lf.nodes.unregister_node(PosterizeExample.id)
