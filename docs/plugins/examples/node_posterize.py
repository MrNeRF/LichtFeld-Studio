"""Example Python node: posterize selected base RGB colours and flatten their SH."""

import lichtfeld as lf


class PosterizeExample(lf.nodes.Node):
    id = "example.posterize"
    label = "Posterize Example"
    category = "Colour"
    description = "Reduces your scene's colours to a small number of flat steps."
    help = "Start with Levels 4 for four steps per colour channel.\nConnect a Selection to limit the effect."

    inputs = [
        lf.nodes.Input("Geometry", "geometry", description="Splats you want to posterise. Connect a source; unconnected means no geometry."),
        lf.nodes.Input("Selection", "float", 1.0, min=0.0, max=1.0, field=True,
                       description="Effect strength, 0–1: 0 keeps the original, 1 fully posterises it. Unconnected means everything."),
        lf.nodes.Input("Levels", "int", 4, min=2, max=32,
                       description="Colour steps per channel, 2–32. 2 gives low and high values; 4 is the default."),
    ]
    outputs = [lf.nodes.Output("Geometry", "geometry", description="Posterised splats, with selected camera-dependent colour faded out.")]

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


def on_load():
    lf.nodes.register_node(PosterizeExample)


def on_unload():
    lf.nodes.unregister_node(PosterizeExample.id)
