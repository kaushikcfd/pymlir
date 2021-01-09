import sys
from mlir import parse_string
from mlir.builder import IRBuilder
from mlir.builder import Reads, Writes, Isa, And
from mlir.dialects.affine import AffineLoadOp


def test_saxpy_builder():
    builder = IRBuilder()
    F64 = builder.F64
    Mref1D = builder.MemRefType(shape=(None, ), dtype=F64)

    mlirfile = builder.make_mlir_file()
    module = mlirfile.module

    with builder.goto_block(builder.make_block(module.region)):
        saxpy_fn = builder.function("saxpy")

    block = builder.make_block(saxpy_fn.region)
    builder.position_at_start(block)

    a, x, y = builder.add_function_args(saxpy_fn, [F64, Mref1D, Mref1D])
    c0 = builder.index_constant(0)
    n = builder.dim(x, c0, builder.INDEX)

    f = builder.affine_for(0, n)
    i = f.index

    with builder.goto_block(builder.make_block(f.region)):
        axi = builder.mulf(builder.affine_load(x, i, Mref1D), a, F64)
        axpyi = builder.addf(builder.affine_load(y, i, Mref1D), axi, F64)
        builder.affine_store(axpyi, y, i, Mref1D)

    builder.ret()

    print(mlirfile.dump())


def test_query():
    block = parse_string("""
func @saxpy(%a : f64, %x : memref<?xf64>, %y : memref<?xf64>) {
%c0 = constant 0 : index
%n = dim %x, %c0 : memref<?xf64>

affine.for %i = 0 to %n {
  %xi = affine.load %x[%i+1] : memref<?xf64>
  %axi =  mulf %a, %xi : f64
  %yi = affine.load %y[%i] : memref<?xf64>
  %axpyi = addf %yi, %axi : f64
  affine.store %axpyi, %y[%i] : memref<?xf64>
}
return
}""").module.region.body[0].region.body[0]
    for_block = block.body[2].op.region.body[0]

    c0 = block.body[0].result_list[0].value

    def query(expr):
        return next((op
                   for op in block.body + for_block.body
                   if expr(op)))

    assert query(Writes("%c0")).dump() == "%c0 = constant 0 : index"
    assert (query(Reads("%y") & Isa(AffineLoadOp)).dump()
            == "%yi = affine.load %y [ %i ] : memref<?xf64>")

    assert query(Reads(c0)).dump() == "%n = dim %x , %c0 : memref<?xf64>"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
