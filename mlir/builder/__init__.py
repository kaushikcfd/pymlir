import mlir.astnodes as ast
import mlir.dialects.standard as std
import mlir.dialects.affine as affine
from typing import Optional, Tuple, Union, List
from pytools import UniqueNameGenerator
from contextlib import contextmanager
from mlir.builder.match import Reads, Writes, Isa, Not, All  # noqa: F401
from mlir.builder.match import MatchExpressionBase


class IRBuilder:
    """
    Mutable class as a helper to build MLIR block AST. The aim of this class is
    to provide the building blocks for the core dialect operations. Builders
    that include custom dialects can sub-class from this and implement support
    for the operations within the custom dialect.

    .. attribute:: block

        The block that the builder is operating on.

    .. attribute:: position

        An instance of :class:`int`, indicating the position where the next
        operation is to be added in the .

    .. note::

        * The concepts here at not true to the implementation in
          llvm-project/mlir. It should be seen more of a convenience to emit MLIR
          modules.

        * This class shared design elements from :class:`llvmlite.ir.IRBuilder`,
          querying mechanism from :mod:`loopy`
    """

    def __init__(self):
        self.block = None
        self.position = None

    name_gen = UniqueNameGenerator(forced_prefix="_pymlir_")

    F16 = ast.FloatType(type=ast.FloatTypeEnum.f16)
    F32 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    F64 = ast.FloatType(type=ast.FloatTypeEnum.f64)
    INT32 = ast.IntegerType(32)
    INT64 = ast.IntegerType(64)
    INDEX = ast.IndexType()

    @staticmethod
    def make_mlir_file(module: Optional[ast.Module] = None) -> ast.MLIRFile:
        if module is None:
            module = ast.Module(None, None, ast.Region([]))
        return ast.MLIRFile([], module)

    def module(self, name: Optional[str] = None) -> ast.Module:
        if name is None:
            name = None
        else:
            name = ast.SymbolRefId(name)

        op = ast.Module(name, None, ast.Region([]))
        self._append_op_to_block([], op)
        return op

    def function(self, name: Optional[str] = None) -> ast.Function:
        if name is None:
            name = self.name_gen("fn")

        op = ast.Function(ast.SymbolRefId(value=name), [], [], None, ast.Region([]))

        self._append_op_to_block([], op)
        return op

    @classmethod
    def make_block(cls, region: ast.Region, name: Optional[str] = None) -> ast.Block:
        if name is None:
            label = None
        else:
            label = ast.BlockLabel(name, [], [])

        block = ast.Block(label, [])
        region.body.append(block)
        return block

    @classmethod
    def add_function_args(cls, function: ast.Function, dtypes: List[ast.Type],
                          names: Optional[List[str]] = None,
                          positions: Optional[List[int]] = None):
        if names is None:
            names = [cls.name_gen("fnarg") for _ in dtypes]

        if function.args is None:
            function.args = []

        if positions is None:
            positions = list(range(len(function.args), len(function.args) + len(dtypes)))

        args = []

        for name, dtype, pos in zip(names, dtypes, positions):
            arg = ast.SsaId(name)
            function.args.insert(pos, ast.NamedArgument(arg, dtype))
            args.append(arg)

        return args

    def MemRefType(self,
                   dtype: ast.Type,
                   shape: Optional[Tuple[Optional[int], ...]],
                   offset: Optional[int] = None,
                   strides: Optional[Tuple[Optional[int], ...]] = None) -> ast.MemRefType:
        """
        If *shape* is None, returns an instance of
        :class:`mlir.astnodes.UnrankedMemRefType`. If *shape* is a
        :class:`tuple`, returns a :class:`mlir.astnodes.RankedMemRefType`.
        """
        if shape is None:
            assert strides is None
            return ast.UnrankedMemRefType(dtype)
        else:
            shape = tuple(ast.Dimension(dim) for dim in shape)
            if strides is None and offset is None:
                layout = None
            else:
                if offset is None:
                    offset = 0
                if strides is not None:
                    if len(shape) != len(strides):
                        raise ValueError("shapes and strides must be of tuples of same dimensionality.")
                layout = ast.StridedLayout(strides, offset)

            return ast.RankedMemRefType(shape, dtype, layout)

    def _append_op_to_block(self, op_results: List[Optional[Union[ast.SsaId, str]]], op):
        new_op_results = []
        for op_result in op_results:
            if op_result is None:
                op_result = self.name_gen("ssa")

            if isinstance(op_result, str):
                result = ast.SsaId(op_result)

            new_op_results.append(result)

        if self.block is None:
            raise ValueError("Not within any block to append")

        self.block.body.append(ast.Operation(result_list=new_op_results, op=op,
                                              location=None))

        if len(new_op_results) == 1:
            return new_op_results[0]
        elif len(new_op_results) > 1:
            return new_op_results
        else:
            return

    def position_at_start(self, block: ast.Block):
        self.block = block
        self.position = 0

    def position_at_end(self, block: ast.Block):
        self.block = block
        self.position = len(block.body)

    @contextmanager
    def goto_block(self, block: ast.Block):
        parent_block = self.block
        parent_position = self.position

        self.position_at_end(block)
        yield

        self.block = parent_block
        self.position = parent_position

    @contextmanager
    def goto_entry_block(self, block: ast.Block):
        parent_block = self.block
        parent_position = self.position

        self.position_at_start(block)
        yield

        self.block = parent_block
        self.position = parent_position

    def position_before(self, query: MatchExpressionBase, block: Optional[ast.Block] = None):
        if block is not None:
            self.block = block

        try:
            self.position = next((i
                                  for i, op in enumerate(self.block.body)
                                  if query(op)))
        except StopIteration:
            raise ValueError(f"Did not find an operation matching '{query}'.")

    def position_after(self, query, block: Optional[ast.Block] = None):
        if block is not None:
            self.block = block

        try:
            self.position = next((i
                                  for i, op in zip(range(len(self.block)-1, -1, -1),
                                                   reversed(self.block.body))
                                  if query(op)))
        except StopIteration:
            raise ValueError(f"Did not find an operation matching '{query}'.")

    @contextmanager
    def goto_before(self, query: MatchExpressionBase, block: Optional[ast.Block] = None):
        parent_block = self.block
        parent_position = self.position

        self.position_before(query, block)
        yield

        self.block = parent_block
        self.position = parent_position

    @contextmanager
    def goto_after(self, query: MatchExpressionBase, block: Optional[ast.Block] = None):
        parent_block = self.block
        parent_position = self.position

        self.position_after(query, block)
        yield

        self.block = parent_block
        self.position = parent_position

    def make_attribute_entry(self, name: str, value: ast.Attribute):
        raise NotImplementedError()

    # {{{ standard dialect

    def addf(self, op_a: ast.SsaId, op_b: ast.SsaId, type: ast.Type,
             name: Optional[str] = None):
        op = std.AddfOperation(_match=0, operand_a=op_a, operand_b=op_b, type=type)
        return self._append_op_to_block([name], op)

    def mulf(self, op_a: ast.SsaId, op_b: ast.SsaId, type: ast.Type,
             name: Optional[str] = None):
        op = std.MulfOperation(_match=0, operand_a=op_a, operand_b=op_b, type=type)
        return self._append_op_to_block([name], op)

    def dim(self, memref_or_tensor: ast.SsaId, index: ast.SsaId,
            memref_type: Union[ast.MemRefType, ast.TensorType],
            name: Optional[str] = None):
        op = std.DimOperation(_match=0, operand=memref_or_tensor, index=index, type=memref_type)
        return self._append_op_to_block([name], op)

    def index_constant(self, value: int, name: Optional[str] = None):
        op = std.ConstantOperation(_match=0, value=value, type=ast.IndexType())
        return self._append_op_to_block([name], op)

    def float_constant(self, value: float, type: ast.FloatType, name: Optional[str] = None):
        op = std.ConstantOperation(_match=0, value=value, type=type)
        return self._append_op_to_block([name], op)

    # }}}

    # {{{ affine dialect

    def affine_for(self, lower_bound: Union[int, ast.SsaId],
                   upper_bound: Union[int, ast.SsaId],
                   step: Optional[int] = None, indexname: Optional[str] = None):
        if indexname is None:
            indexname = self.name_gen("i")
            index = ast.AffineSsa(indexname)

        if step is None:
            match = 0
        else:
            match = 1

        op = affine.AffineForOp(_match=match,
                                index=index,
                                begin=lower_bound, end=upper_bound, step=step,
                                region=ast.Region(body=[]))

        self._append_op_to_block([], op)
        return op

    def affine_load(self, memref: ast.SsaId, indices: Union[ast.AffineExpr, List[ast.AffineExpr]],
                    memref_type: ast.MemRefType, name: Optional[str] = None):
        if isinstance(indices, ast.AffineExpr):
            indices = [indices]

        op = affine.AffineLoadOp(_match=0, arg=memref, index=ast.MultiDimAffineExpr(indices), type=memref_type)
        return self._append_op_to_block([name], op)

    def affine_store(self, address: ast.SsaId, memref: ast.SsaId,
                     indices: Union[ast.AffineExpr, List[ast.AffineExpr]],
                     memref_type: ast.MemRefType):
        if isinstance(indices, ast.AffineExpr):
            indices = [indices]

        op = affine.AffineStoreOp(_match=0, addr=address, ref=memref,
                                  index=ast.MultiDimAffineExpr(indices), type=memref_type)
        self._append_op_to_block([], op)

    def ret(self, values: Optional[List[ast.SsaId]] = None,
            types: Optional[List[ast.Type]] = None):

        op = std.ReturnOperation(_match=0, values=values, types=types)
        self._append_op_to_block([], op)
        self.block = None
        self.position = 0

    # }}}


# vim: fdm=marker
