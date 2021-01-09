""" MLIR IR Builder."""

__copyright__ = "Copyright (C) 2020 Kaushik Kulkarni"

__license__ = """
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import mlir.astnodes as ast
import mlir.dialects.standard as std
import mlir.dialects.affine as affine
from typing import Optional, Tuple, Union, List
from pytools import UniqueNameGenerator
from contextlib import contextmanager
from mlir.builder.match import Reads, Writes, Isa, All, And, Or, Not  # noqa: F401
from mlir.builder.match import MatchExpressionBase


__doc__ = """
.. currentmodule:: mlir.builder

.. autoclass:: IRBuilder

.. automodule:: mlir.builder.match
"""


class IRBuilder:
    """
    MLIR AST builder. Provides convenience methods for adding core dialect
    operations to a :class:`~mlir.astnodes.Block`.

    .. attribute:: block

        The block that the builder is operating on.

    .. attribute:: position

        An instance of :class:`int`, indicating the position where the next
        operation is to be added in the :attr:`block`.

    .. note::

        * The concepts here at not true to the implementation in
          llvm-project/mlir. It should be seen more of a convenience to emit MLIR
          modules.

        * This class shared design elements from :class:`llvmlite.ir.IRBuilder`,
          querying mechanism from :mod:`loopy`.

    *Position/block manipulation*

    .. automethod:: position_at_entry
    .. automethod:: position_at_exit
    .. automethod:: goto_block
    .. automethod:: goto_entry_block
    .. automethod:: goto_before
    .. automethod:: goto_after

    *Types*

    :attr F16: f16 type
    :attr F32: f32 type
    :attr F64: f64 type
    :attr INT32: i32 type
    :attr INT64: i64 type
    :attr INDEX: index type

    .. automethod:: MemRefType

    *Affine dialect ops*

    .. automethod:: affine_for
    .. automethod:: affine_load
    .. automethod:: affine_store

    *Standard dialect ops*

    .. automethod:: dim
    .. automethod:: addf
    .. automethod:: mulf
    .. automethod:: index_constant
    .. automethod:: float_constant
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
        """
        Returns an instance of :class:`mlir.astnodes.MLIRFile` for *module*.
        If *module* is *None*, defaults it with an empty :class:`mlir.astnodes.Module`.
        """
        if module is None:
            module = ast.Module(None, None, ast.Region([]))
        return ast.MLIRFile([], module)

    def module(self, name: Optional[str] = None) -> ast.Module:
        """
        Inserts a :class:`mlir.astnodes.Module` with name *name* into *block*.

        Returns the inserted module.
        """
        if name is None:
            name = None
        else:
            name = ast.SymbolRefId(name)

        op = ast.Module(name, None, ast.Region([]))
        self._insert_op_in_block([], op)
        return op

    def function(self, name: Optional[str] = None) -> ast.Function:
        """
        Inserts a :class:`mlir.astnodes.Function` with name *name* into *block*.

        Returns the inserted function.
        """
        if name is None:
            name = self.name_gen("fn")

        op = ast.Function(ast.SymbolRefId(value=name), [], [], None, ast.Region([]))

        self._insert_op_in_block([], op)
        return op

    @classmethod
    def make_block(cls, region: ast.Region, name: Optional[str] = None) -> ast.Block:
        """
        Appends a :class:`mlir.astnodes.Block` with name *name* to the *region*.

        Returns the appended block.
        """
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
        """
        Adds arguments to *function*.

        :arg dtypes: Types of the arguments to be added to the function.
        :arg names: Names of the arguments to be added to the function.
        :arg positions: Positions where the arguments are to be inserted.
        """
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
        Returns an instance of :class:`mlir.astnodes.UnrankedMemRefType` if shape is
        *None*, else returns a :class:`mlir.astnodes.RankedMemRefType`.
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

    def _insert_op_in_block(self, op_results: List[Optional[Union[ast.SsaId, str]]], op):
        new_op_results = []
        for op_result in op_results:
            if op_result is None:
                op_result = self.name_gen("ssa")

            if isinstance(op_result, str):
                result = ast.SsaId(op_result)

            new_op_results.append(result)

        if self.block is None:
            raise ValueError("Not within any block to append")

        self.block.body.insert(self.position, ast.Operation(result_list=new_op_results,
                                                            op=op))
        self.position += 1

        if len(new_op_results) == 1:
            return new_op_results[0]
        elif len(new_op_results) > 1:
            return new_op_results
        else:
            return

    # {{{ position/block manipulation

    def position_at_entry(self, block: ast.Block):
        """
        Starts building at *block*'s entry.
        """
        self.block = block
        self.position = 0

    def position_at_exit(self, block: ast.Block):
        """
        Starts building at *block*'s exit.
        """
        self.block = block
        self.position = len(block.body)

    @contextmanager
    def goto_block(self, block: ast.Block):
        """
        Context to start building at *block*'s exit.

        Example usage::

            with builder.goto_block(block):
                # starts building at *block*'s exit.
                z = builder.addf(x, y, F64)

            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_at_exit(block)
        yield

        self.block = parent_block
        self.position = parent_position

    @contextmanager
    def goto_entry_block(self, block: ast.Block):
        """
        Context to start building at *block*'s entry.

        Example usage::

            with builder.goto_block(block):
                # starts building at *block*'s entry.
                z = builder.addf(x, y, F64)

            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_at_entry(block)
        yield

        self.block = parent_block
        self.position = parent_position

    def position_before(self, query: MatchExpressionBase, block: Optional[ast.Block] = None):
        """
        Positions the builder to the point just before *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            builder.position_before(Reads("%c0") & Isa(AddfOperation))
            # starts building before operation of form "... = addf %c0, ..."
        """
        if block is not None:
            self.block = block

        try:
            self.position = next((i
                                  for i, op in enumerate(self.block.body)
                                  if query(op)))
        except StopIteration:
            raise ValueError(f"Did not find an operation matching '{query}'.")

    def position_after(self, query, block: Optional[ast.Block] = None):
        """
        Positions the builder to the point just after *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            builder.position_after(Writes("%c0") & Isa(ConstantOperation))
            # starts building after operation of form "%c0 = constant ...: ..."
        """
        if block is not None:
            self.block = block

        try:
            self.position = next((i
                                  for i, op in zip(range(len(self.block.body)-1, -1, -1),
                                                   reversed(self.block.body))
                                  if query(op))) + 1
        except StopIteration:
            raise ValueError(f"Did not find an operation matching '{query}'.")

    @contextmanager
    def goto_before(self, query: MatchExpressionBase, block: Optional[ast.Block] = None):
        """
        Enters a context to build at the point just before *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            with builder.goto_before(Reads("%c0") & Isa(AddfOperation)):
                # starts building before operation of form "... = addf %c0, ..."
                z = builder.mulf(x, y, F64)
            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_before(query, block)

        entered_at = self.position
        yield

        exit_at = self.position
        self.block = parent_block

        # accounting for operations added within the context
        if entered_at <= parent_position:
            parent_position += (exit_at - entered_at)

        self.position = parent_position + (exit_at - entered_at)

    @contextmanager
    def goto_after(self, query: MatchExpressionBase, block: Optional[ast.Block] = None):
        """
        Enters a context to build at the point just after *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            with builder.goto_after(Writes("%c0") & Isa(ConstantOperation)):
                # starts building after operation of form "%c0 = constant ...: ..."
                z = builder.dim(x, c0, builder.INDEX)

            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_after(query, block)

        entered_at = self.position
        yield

        exit_at = self.position
        self.block = parent_block

        # accounting for operations added within the context
        if entered_at <= parent_position:
            parent_position += (exit_at - entered_at)

        self.position = parent_position

    # }}}

    # {{{ standard dialect

    def addf(self, op_a: ast.SsaId, op_b: ast.SsaId, type: ast.Type,
             name: Optional[str] = None):
        op = std.AddfOperation(_match=0, operand_a=op_a, operand_b=op_b, type=type)
        return self._insert_op_in_block([name], op)

    def mulf(self, op_a: ast.SsaId, op_b: ast.SsaId, type: ast.Type,
             name: Optional[str] = None):
        op = std.MulfOperation(_match=0, operand_a=op_a, operand_b=op_b, type=type)
        return self._insert_op_in_block([name], op)

    def dim(self, memref_or_tensor: ast.SsaId, index: ast.SsaId,
            memref_type: Union[ast.MemRefType, ast.TensorType],
            name: Optional[str] = None):
        op = std.DimOperation(_match=0, operand=memref_or_tensor, index=index, type=memref_type)
        return self._insert_op_in_block([name], op)

    def index_constant(self, value: int, name: Optional[str] = None):
        op = std.ConstantOperation(_match=0, value=value, type=ast.IndexType())
        return self._insert_op_in_block([name], op)

    def float_constant(self, value: float, type: ast.FloatType, name: Optional[str] = None):
        op = std.ConstantOperation(_match=0, value=value, type=type)
        return self._insert_op_in_block([name], op)

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

        self._insert_op_in_block([], op)
        return op

    def affine_load(self, memref: ast.SsaId, indices: Union[ast.AffineExpr, List[ast.AffineExpr]],
                    memref_type: ast.MemRefType, name: Optional[str] = None):
        if isinstance(indices, ast.AffineExpr):
            indices = [indices]

        op = affine.AffineLoadOp(_match=0, arg=memref, index=ast.MultiDimAffineExpr(indices), type=memref_type)
        return self._insert_op_in_block([name], op)

    def affine_store(self, address: ast.SsaId, memref: ast.SsaId,
                     indices: Union[ast.AffineExpr, List[ast.AffineExpr]],
                     memref_type: ast.MemRefType):
        if isinstance(indices, ast.AffineExpr):
            indices = [indices]

        op = affine.AffineStoreOp(_match=0, addr=address, ref=memref,
                                  index=ast.MultiDimAffineExpr(indices), type=memref_type)
        self._insert_op_in_block([], op)

    def ret(self, values: Optional[List[ast.SsaId]] = None,
            types: Optional[List[ast.Type]] = None):

        op = std.ReturnOperation(_match=0, values=values, types=types)
        self._insert_op_in_block([], op)
        self.block = None
        self.position = 0

    # }}}


# vim: fdm=marker
