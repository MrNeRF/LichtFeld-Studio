"""GDB/cuda-gdb pretty printers for the LichtFeld tensor library.

Load with:  source tools/gdb/lfs_pretty_printers.py
"""

import gdb
import gdb.printing

DTYPE_NAMES = {
    0: "float32",
    1: "float16",
    2: "int32",
    3: "int64",
    4: "uint8",
    5: "bool",
}

DTYPE_SIZES = {0: 4, 1: 2, 2: 4, 3: 8, 4: 1, 5: 1}

DTYPE_GDB_TYPES = {
    0: "float",
    2: "int",
    3: "long",
    4: "unsigned char",
    5: "bool",
}

DEVICE_NAMES = {0: "cpu", 1: "cuda"}


def _in_cuda_gdb():
    """True only under cuda-gdb. `info cuda devices` does not exist in plain gdb."""
    try:
        gdb.execute("info cuda devices", to_string=True)
        return True
    except gdb.error:
        return False


def _enum_value(val):
    """Underlying integer of a scoped enum, tolerating gdb's enum rendering."""
    try:
        return int(val)
    except gdb.error:
        return int(val.cast(gdb.lookup_type("unsigned char")))


def _vector_elements(vec, limit=64):
    """Read a std::vector<size_t> without depending on the libstdc++ printers."""
    try:
        impl = vec["_M_impl"]
    except gdb.error:
        return None
    try:
        start = impl["_M_start"]
        finish = impl["_M_finish"]
    except gdb.error:
        try:
            start = impl["_M_impl_data"]["_M_start"]
            finish = impl["_M_impl_data"]["_M_finish"]
        except gdb.error:
            return None
    if int(start) == 0:
        return []
    count = int(finish - start)
    if count < 0 or count > limit:
        return None
    return [int((start + i).dereference()) for i in range(count)]


def _shape_dims(shape_val):
    return _vector_elements(shape_val["dims_"])


def _fmt_dims(dims):
    if dims is None:
        return "?"
    return "[" + ", ".join(str(d) for d in dims) + "]"


class TensorShapePrinter:
    """lfs::core::TensorShape"""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        dims = _shape_dims(self.val)
        try:
            total = int(self.val["total_elements_"])
        except gdb.error:
            total = -1
        rank = len(dims) if dims is not None else "?"
        return f"TensorShape {_fmt_dims(dims)} rank={rank} elems={total}"


class TensorPrinter:
    """lfs::core::Tensor"""

    def __init__(self, val):
        self.val = val

    def _dtype(self):
        try:
            return _enum_value(self.val["dtype_"])
        except gdb.error:
            return None

    def _device(self):
        try:
            return _enum_value(self.val["device_"])
        except gdb.error:
            return None

    def to_string(self):
        dtype = self._dtype()
        device = self._device()
        dtype_name = DTYPE_NAMES.get(dtype, f"dtype{dtype}")
        device_name = DEVICE_NAMES.get(device, f"device{device}")

        try:
            data = int(self.val["data_"])
        except gdb.error:
            data = 0

        if data == 0:
            return f"Tensor <empty> {dtype_name} {device_name}"

        dims = _shape_dims(self.val["shape_"])
        try:
            total = int(self.val["shape_"]["total_elements_"])
        except gdb.error:
            total = -1

        flags = []
        try:
            if not bool(self.val["is_contiguous_"]):
                flags.append("strided")
        except gdb.error:
            pass
        try:
            if bool(self.val["is_view_"]):
                flags.append("view")
        except gdb.error:
            pass
        try:
            off = int(self.val["storage_offset_"])
            if off:
                flags.append(f"offset={off}")
        except gdb.error:
            pass

        nbytes = total * DTYPE_SIZES.get(dtype, 0) if total > 0 else 0
        suffix = (" " + " ".join(flags)) if flags else ""
        return (
            f"Tensor {dtype_name}{_fmt_dims(dims)} {device_name} "
            f"elems={total} bytes={nbytes} data=0x{data:x}{suffix}"
        )

    def children(self):
        """Readable members. strides/dtype/device are rendered here rather than
        handed to gdb, so output does not depend on the libstdc++ printers being
        auto-loaded (they are blocked by default under `gdb -nx`)."""
        dims = _shape_dims(self.val["shape_"])
        yield "shape", _fmt_dims(dims)

        strides = _vector_elements(self.val["strides_"])
        yield "strides", _fmt_dims(strides)

        dtype = self._dtype()
        yield "dtype", DTYPE_NAMES.get(dtype, str(dtype))
        device = self._device()
        yield "device", DEVICE_NAMES.get(device, str(device))

        for label, name in (
            ("storage_offset", "storage_offset_"),
            ("is_contiguous", "is_contiguous_"),
            ("is_view", "is_view_"),
            ("data", "data_"),
            ("id", "id_"),
        ):
            try:
                yield label, self.val[name]
            except gdb.error:
                continue


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("lfs")
    pp.add_printer("lfs::core::Tensor", "^lfs::core::Tensor$", TensorPrinter)
    pp.add_printer("lfs::core::TensorShape", "^lfs::core::TensorShape$", TensorShapePrinter)
    return pp


class LfsTensorDump(gdb.Command):
    """Dump tensor element values: lfs-tensor <expr> [count]

    CPU tensors read directly. CUDA tensors require cuda-gdb; under plain gdb the
    device pointer is unreadable and the command says so instead of printing junk.
    """

    def __init__(self):
        super().__init__("lfs-tensor", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        argv = gdb.string_to_argv(arg)
        if not argv:
            raise gdb.GdbError("usage: lfs-tensor <expr> [count]")
        count = int(argv[1]) if len(argv) > 1 else 16

        val = gdb.parse_and_eval(argv[0])
        if val.type.code == gdb.TYPE_CODE_PTR:
            val = val.dereference()

        dtype = _enum_value(val["dtype_"])
        device = _enum_value(val["device_"])
        data = int(val["data_"])
        if data == 0:
            print("tensor has no storage")
            return

        dims = _shape_dims(val["shape_"])
        total = int(val["shape_"]["total_elements_"])
        offset = int(val["storage_offset_"])
        gdb_type = DTYPE_GDB_TYPES.get(dtype)

        print(
            f"{DTYPE_NAMES.get(dtype)}{_fmt_dims(dims)} on "
            f"{DEVICE_NAMES.get(device)}, {total} elems, offset={offset}"
        )
        if gdb_type is None:
            print(f"no host read-back for dtype {dtype} (float16 needs conversion)")
            return

        n = min(count, total)
        addr = data + offset * DTYPE_SIZES.get(dtype, 1)

        if device == 1:
            # A device pointer is usually still *readable* through an unqualified
            # cast -- it silently returns zeros, in both gdb and cuda-gdb. Reading
            # it correctly requires cuda-gdb's @global address-space qualifier.
            if not _in_cuda_gdb():
                print(
                    f"refusing to read CUDA memory at 0x{data:x} under plain gdb: "
                    "an unqualified read appears to succeed and returns garbage. "
                    "Use cuda-gdb."
                )
                return
            try:
                arr = gdb.parse_and_eval(f"*(@global {gdb_type} (*)[{n}])({addr})")
                vals = [arr[i] for i in range(n)]
            except gdb.error as exc:
                print(f"device read failed: {exc}")
                return
        else:
            try:
                ptr = gdb.Value(addr).cast(gdb.lookup_type(gdb_type).pointer())
                vals = [ptr[i] for i in range(n)]
            except gdb.MemoryError:
                print(f"cannot read host memory at 0x{addr:x}")
                return

        print("  " + ", ".join(str(v) for v in vals) + (" ..." if n < total else ""))


def register(objfile=None):
    gdb.printing.register_pretty_printer(
        objfile if objfile is not None else gdb.current_objfile(),
        build_pretty_printer(),
        replace=True,
    )


register(gdb.current_objfile() or gdb)
LfsTensorDump()
print("[lfs] tensor pretty printers loaded (lfs-tensor <expr> [count] to dump values)")
