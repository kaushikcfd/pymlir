"""Operation query interface."""

import mlir.astnodes as ast
from mlir.visitor import NodeVisitor
from typing import List, Union
from dataclasses import dataclass


class SsaIdCollector(NodeVisitor):
    def __init__(self):
        self.visited_ssas = set()

    def visit_SsaId(self, ssa: ast.SsaId):
        self.visited_ssas.add(ssa)


class MatchExpressionBase:
    def __call__(self, op: ast.Operation) -> bool:
        raise NotImplementedError()

    def __and__(self, other: "MatchExpressionBase") -> "And":
        return And([self, other])

    def __or__(self, other: "MatchExpressionBase") -> "Or":
        return Or([self, other])


class All(MatchExpressionBase):
    def __call__(self, op: ast.Operation) -> bool:
        return True


@dataclass
class And(MatchExpressionBase):
    children: List[MatchExpressionBase]

    def __call__(self, op: ast.Operation) -> bool:
        return all(ch(op) for ch in self.children)


@dataclass
class Or(MatchExpressionBase):
    children: List[MatchExpressionBase]

    def __call__(self, op: ast.Operation) -> bool:
        return any(ch(op) for ch in self.children)


@dataclass
class Not(MatchExpressionBase):
    child: MatchExpressionBase

    def __call__(self, op: ast.Operation) -> bool:
        return not self.child(op)


@dataclass
class Reads(MatchExpressionBase):
    name: Union[str, ast.SsaId]

    def __call__(self, op: ast.Operation) -> bool:
        visited_ssas = SsaIdCollector().visit(op.operation).visited_ssas
        return any((ssa is self.name) or (ssa.dump() == self.name)
                   for ssa in visited_ssas)


@dataclass
class Writes(MatchExpressionBase):
    name: Union[str, ast.SsaId]

    def __call__(self, op: ast.Operation) -> bool:
        return any((ssa is self.name) or (ssa.dump() == self.name)
                   for ssa in op.result_list)


@dataclass
class Isa(MatchExpressionBase):
    type: type

    def __call__(self, op: ast.Operation) -> bool:
        return isinstance(op.operation, self.type)


writes = Writes
reads = Reads
isa = Isa
not_ = Not

# vim: fdm=marker
