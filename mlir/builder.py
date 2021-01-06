import mlir.astnodess as ast
import mlir.dialects.standard as std
import mlir.dialects.affine as affine
from typing import Optional, Tuple, Union, List
from pytools import UniqueNameGenerator


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

    .. attribute:: functions

        A mapping from function names to the :class:`ast.Function` of
        :attr:`module`.

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

    F16 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    F32 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    F64 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    INT32 = ast.IntegerType(32)
    INT64 = ast.IntegerType(64)
    INDEX = ast.IndexType()

    @staticmethod
    def make_mlir_file(module: ast.Module) -> ast.MLIRFile:
        return ast.MLIRFile([], module)

    @classmethod
    def make_module(cls, name: Optional[str]) -> ast.Module:
        if name is not None:
            # FIXME: aise hi kuch lenga module.
            name = cls.name_gen("fn")
        return ast.Module(ast.SymbolRefId(name))

    @classmethod
    def make_function(cls, name: Optional[str] = None) -> ast.Function:
        if name is None:
            name = cls.name_gen("fn")

        return ast.Function(name=ast.SymbolRefId(value=name))

    @classmethod
    def make_block(cls, region: ast.Reagion, name: Optional[str]) -> ast.Block:
        if name is None:
            name = cls.name_gen("bb")

        raise NotImplementedError()

        block = ast.Block(...)
        region.body.append(block)
        return block

    def MemRefType(self,
                   name: str,
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
            if len(shape) != len(strides):
                raise ValueError("shapes and strides must be of tuples of same dimensionality.")

            if strides is None and offset is None:
                layout = None
            else:
                if offset is None:
                    offset = 0
                layout = ast.StridedLayout(strides, offset)

            return ast.RankedMemRefType(dimensions=shape,
                    element_type=dtype, memory_space=None, layout=layout)

    def _append_op_to_block(self, op_results, op):
        if self.block is None:
            raise ValueError("Not within any block to append")

        self.block.body.append(ast.Operation(result_list=op_results, op=op,
                                              location=None))

    def position_before(self, query):
        # would need some traversal to get this right.
        # I'm imagining this design would be similar to loopy's query language.
        # As: builder.position_before("reads:%a and"
        #                             " isa(mlir.dialects.affine.AffineStore))"
        raise NotImplementedError()

    def position_after(self, query):
        # would need some traversal to get this right.
        raise NotImplementedError()

    def position_at_start(self, block: ast.Block):
        self.position = 0
        self.block = block

    def position_at_end(self, block: ast.Block):
        self.position = len(block.body)
        self.block = block

    def make_attribute_entry(self, name: str, value: ast.Attribute):
        raise NotImplementedError()

    def addf(self, op_a: ast.SsaId, op_b: ast.SsaId, type: ast.Type,
             name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder
        op = std.Addf(operand_a=op_a, operand_b=op_b, type=type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_block([result], op)

        return result

    def mulf(self, op_a: ast.SsaId, op_b: ast.SsaId, ret_type: ast.Type,
             name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder
        op = std.Mulf(operand_a=op_a, operand_b=op_b, type=type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_block([result], op)

        return result

    def return_from_function(self):
        raise NotImplementedError()

    def dim(self, memref_or_tensor: ast.SsaId, index: ast.SsaId,
            memref_type: Union[ast.MemRefType, ast.TensorType],
            name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder
        op = std.DimOperation(operand=memref_or_tensor, index=index, type=memref_type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_block([result], op)

        return result

    def index_constant(self, value: int, name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder

        op = std.ConstantOperation(value=ast.IntegerAttribute(value),
                                               type=ast.IndexType())
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_block([result], op)

        return result

    def float_constant(self, value: float, name: Optional[str], type: ast.FloatType):
        # TODO: This should be defined by StdIRBuilder

        op = std.ConstantOperation(value=ast.FloatAttribute(value),
                                               type=type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_block([result], op)

        return result

    def affine_for(self, lower_bound: Union[int, ast.SsaId],
                   upper_bound: Union[int, ast.SsaId],
                   step: Optional[int] = None, indexname: Optional[str] = None):
        #TODO: This should be defined by AffineIRBuilder
        parent_block = self.block
        parent_position = self.position

        if indexname is None:
            indexname = self.name_gen("i")
            index = ast.AffineSsa(value=indexname, index=None)

        op = affine.AffineForOp(begin=lower_bound, end=upper_bound, step=step,
                region=ast.Region(body=[]), index=index)
        self._append_op_to_block([], op)

        self.block = None
        self.position = None
        yield op

        self.block = parent_block
        self.position = parent_position

    def affine_load(self, memref: ast.SsaId, indices: List[ast.AffineNode],
            memref_type: ast.MemRefType, name=Optional[str]):
        op = affine.AffineLoadOp(arg=memref, index=ast.MultiDimAffineExpression(indices), type=memref_type)

        if name is None:
            name = self.name_gen("ssa")

        result = ast.SsaId(value=name)

        self._append_op_to_block([result], op)

        return result
