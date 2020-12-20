import mlir.astnodess as ast
import mlir.dialects.standard as std
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pytools import Uniqeu


@dataclass(init=True)
class SymbolProperties:
    dtype: ast.Type


class IRBuilder:
    """
    Mutable class as a helper to build MLIR.
    """
    def __init__(self):
        self.current_module = None
        self.current_function = None
        self.current_block = None

        self.name_gen = pytools.UniqueNameGenerator(forced_prefix="_pymlir_")

    F16 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    F32 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    F64 = ast.FloatType(type=ast.FloatTypeEnum.f32)
    Int32 = ast.IntegerType(32)
    Int64 = ast.IntegerType(64)

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


        ...

    def make_module(self, name: str) -> ast.Module:
        if self.current_module is not None:
            raise ValueError("Cannot overwrite a module, instantiate a new"
                             " IRBuilder")

        self.current_module = ast.Module.from_fields(...)
        return self.current_module

    def make_function(self, name: Optiona[str] = None) -> ast.Function:
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

    def make_attribute_entry(self, name: str, value: ast.Attribute):
        raise NotImplementedError()

    def add_function_arg(self, dtype: ast.Type, name: Optional[str] = None):
        if name is None:
            name = self.name_gen("fnarg")

        if self.current_function is None:
            raise ValueError("Not within a function to add args to it.")

        fn = self.current_function
        args = fn.args
        args.append(ast.NamedArgument.from_fields(ast.SsaId(value=name), type=dtype))

    def addf(self, op_a: ast.SsaId, op_b: ast.SsaId, type: ast.Type,
             name: Optional[str]):
        op = std.Addf.from_fields(operand_a=op_a, operand_b=op_b, type=type)
        if name is None:
            name = self.name_gen("ssa")
            result = ast.SsaId(value=name)

        self._append_op_to_current_block(result, op)

        return result

    def mulf(self, op_a: ast.SsaId, op_b: ast.SsaId, ret_type: ast.Type):
        ...

    def dim(self, ...):
        ...

    def constant(self, ...):
        ...

    def affine_store(self, ...):
        ...

    def affine_load(self, ...):
        ...
