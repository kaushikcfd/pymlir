import mlir.astnodess as ast
import mlir.dialects.standard as std
from typing import Optional, Tuple, Union
from pytools import UniqueNameGenerator


class IRBuilder:
    """
    Mutable class as a helper to build MLIR AST. The aim of this class is to
    provide the building blocks for the core dialect operations. Builders that
    include custom dialects can sub-class from this and implement support for
    the operations within the custom dialect.

    .. attribute:: module

        The module that the builder is operating on.

    .. attribute:: function

        The function that the builder is operating on.

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
        self.module = None
        self.function = None
        self.block = None
        self.functions = {}

        self.position = 0

        self.name_gen = UniqueNameGenerator(forced_prefix="_pymlir_")

    F16 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    F32 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    F64 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    INT32 = ast.IntegerType(32)
    INT64 = ast.IntegerType(64)
    INDEX = ast.IndexType

    def MemRefType(self,
                   name: str,
                   dtype: ast.Type,
                   shape: Optional[Tuple[Optional[int], ...]],
                   offset: Optional[int] = None,
                   strides: Optional[Tuple[Optional[int], ...]] = None):
        if shape is None:
            assert strides is None
            return ast.UnrankedMemRefType.from_fields(dtype)
        else:
            return ast.RankedMemRefType.from_fields(...)

    def _append_op_to_current_block(self, op_result, op):
        if self.current_function is None:
            raise ValueError("Not within a function to add args to it.")

        if self.current_block is None:
            self.current_block = ast.Block(label=None)
            fnbody = self.current_function.body.body
            fnbody.append(self.current_block)

        body = self.current_block.body
        body.append(ast.Operation.from_fields(result_list=[op_result], op=op,
                                              location=None))

    def make_module(self, name: str) -> ast.Module:
        if self.current_module is not None:
            raise ValueError("Cannot overwrite a module, instantiate a new"
                             " IRBuilder")

        self.current_module = ast.Module.from_fields(...)
        return self.current_module

    def make_function(self, name: Optional[str] = None) -> ast.Function:
        if self.current_function is not None:
            raise ValueError(f"Already inside a function {self.current_function.name}"
                             f". Must return from it, before creating another.")

        if name is None:
            name = self.name_gen("fn")

        self.current_function = ast.Function.from_fields(name=ast.SymbolRefId.from_fields(value=name))
        return self.current_function

    def make_block(self, name: str) -> ast.Block:
        self.current_block = ast.Block.from_fields(...)
        return self.current_block

    def position_before(self, query):
        # would need some traversal to get this right.
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

    def add_function_arg(self, dtype: ast.Type, name: Optional[str] = None,
                         pos: Optional[int] = None):
        if name is None:
            name = self.name_gen("fnarg")

        if self.current_function is None:
            raise ValueError("Not within a function to add args to it.")

        if pos is None:
            pos = len(self.function.args)

        self.function.args.insert(pos, ast.NamedArgument.from_fields(ast.SsaId(value=name), type=dtype))

    def addf(self, op_a: ast.SsaId, op_b: ast.SsaId, type: ast.Type,
             name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder
        op = std.Addf.from_fields(operand_a=op_a, operand_b=op_b, type=type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_current_block(result, op)

        return result

    def mulf(self, op_a: ast.SsaId, op_b: ast.SsaId, ret_type: ast.Type,
             name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder
        op = std.Mulf.from_fields(operand_a=op_a, operand_b=op_b, type=type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_current_block(result, op)

        return result

    def return_from_function(self):
        raise NotImplementedError()
        self.function = None
        self.block = None

    def dim(self, memref_or_tensor: ast.SsaId, index: ast.SsaId,
            memref_type: Union[ast.MemRefType, ast.TensorType],
            name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder
        op = std.DimOperation.from_fields(operand=memref_or_tensor, index=index, type=memref_type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_current_block(result, op)

        return result

    def index_constant(self, value: int, name: Optional[str]):
        # TODO: This should be defined by StdIRBuilder

        op = std.ConstantOperation.from_fields(value=ast.IntegerAttribute(value),
                                               type=ast.IndexType())
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_current_block(result, op)

        return result
