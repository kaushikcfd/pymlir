import sys
from mlir.builder import IRBuilder


def test_saxpy_builder():
    builder = IRBuilder()
    F64 = builder.F64
    OneDMemref = builder.MemRefType(shape=(None, ), dtype=F64)

    mlirfile = builder.make_mlir_file()
    module = mlirfile.module

    with builder.goto_block(builder.make_block(module.region)):
        saxpy_fn = builder.function("saxpy")

    block = builder.make_block(saxpy_fn.region)
    builder.position_at_start(block)

    a = builder.add_function_arg(saxpy_fn, F64)
    x = builder.add_function_arg(saxpy_fn, OneDMemref)
    y = builder.add_function_arg(saxpy_fn, OneDMemref)
    c0 = builder.index_constant(0)
    n = builder.dim(x, c0, builder.INDEX)

    f = builder.affine_for(0, n)
    i = f.index

    with builder.goto_block(builder.make_block(f.region)):
        axi = builder.mulf(builder.affine_load(x, i, OneDMemref), a, F64)
        axpyi = builder.addf(builder.affine_load(y, i, OneDMemref), axi, F64)
        builder.affine_store(axpyi, y, i, OneDMemref)

    builder.ret()

    mlirfile = builder.make_mlir_file(module)

    print(mlirfile.dump())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
