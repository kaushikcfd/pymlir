""" Implementation of the Linalg dialect. """

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


import inspect
import sys
import mlir.astnodes as ast
from mlir.dialect import Dialect, DialectOp, is_op
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LinalgBatchMatmul(DialectOp):
    a_id: ast.SsaId
    b_id: ast.SsaId
    a_type: ast.Type
    b_type: ast.Type
    c_id: ast.SsaId
    c_type: ast.Type
    out_type: Optional[ast.Type] = None

    _syntax_ = [("linalg.batch_matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_id} : {c_type.type} )"),
                ("linalg.batch_matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " init( {c_id.ssa_id} : {c_type.type} ) -> {out_type.type}")]


@dataclass
class LinalgConvW(DialectOp):
    in_id: ast.SsaId
    filter_id: ast.SsaId
    in_type: ast.Type
    filter_type: ast.Type
    out_id: ast.SsaId
    out_type: ast.Type

    _syntax_ = [("linalg.conv_1d"
                 " ins( {in_id.ssa_id} , {filter_id.ssa_id} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgConvHW(DialectOp):
    in_id: ast.SsaId
    filter_id: ast.SsaId
    in_type: ast.Type
    filter_type: ast.Type
    out_id: ast.SsaId
    out_type: ast.Type

    _syntax_ = [("linalg.conv_2d"
                 " ins( {in_id.ssa_id} , {filter_id.ssa_id} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgConvDHW(DialectOp):
    in_id: ast.SsaId
    filter_id: ast.SsaId
    in_type: ast.Type
    filter_type: ast.Type
    out_id: ast.SsaId
    out_type: ast.Type

    _syntax_ = [("linalg.conv_3d"
                 " ins( {in_id.ssa_id} , {filter_id.ssa_id} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgConv(DialectOp):
    in_id: ast.SsaId
    filter_id: ast.SsaId
    in_type: ast.Type
    filter_type: ast.Type
    out_id: ast.SsaId
    out_type: ast.Type
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.conv( {in_id.ssa_id} , {filter_id.ssa_id} , {out_id.ssa_id} ) "
                "{attr.attribute_value} : {in_type.type} , {filter_type.type} , {out_type.type}"),
                ("linalg.conv( {in_id.ssa_id} , {filter_id.ssa_id} , {out_id.ssa_id} ) "
                " : {in_type.type} , {filter_type.type} , {out_type.type}")]


@dataclass
class LinalgCopy(DialectOp):
    a_id: ast.SsaId
    b_id: ast.SsaId
    a_type: ast.Type
    b_type: ast.Type
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.copy( {a_id.ssa_id} , {b_id.ssa_id} ) "
                "{attr.attribute_value} : {a_type.type} , {b_type.type}"),
                ("linalg.copy( {a_id.ssa_id} , {b_id.ssa_id} ) "
                " : {a_type.type} , {b_type.type}")]


@dataclass
class LinalgDot(DialectOp):
    in_a_id: ast.SsaId
    in_b_id: ast.SsaId
    in_a_type: ast.Type
    in_b_type: ast.Type
    out_id: ast.SsaId
    out_type: ast.Type

    _syntax_ = [("linalg.dot"
                 " ins( {in_a_id.ssa_id} , {in_b_id.ssa_id} : {in_a_type.type} , {in_b_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgFill(DialectOp):
    output_id: ast.SsaId
    value_id: ast.SsaId
    output_type: ast.Type
    value_type: ast.Type
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.fill( {output_id.ssa_id} , {value_id.ssa_id} ) "
                "{attr.attribute_value} : {output_type.type} , {value_type.type}"),
                ("linalg.fill( {output_id.ssa_id} , {value_id.ssa_id} ) "
                " : {output_type.type} , {value_type.type}")]


@dataclass
class LinalgGeneric(DialectOp):
    inargs: List[ast.SsaId]
    in_types: List[ast.Type]
    region: ast.Region
    outargs: Optional[List[ast.SsaId]] = None
    out_types: Optional[List[ast.Type]] = None
    init_args: Optional[List[ast.SsaId]] = None
    init_types: Optional[List[ast.Type]] = None
    out_type: Optional[ast.Type] = None
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " outs( {outargs.ssa_id_list} : {out_types.type_list_no_parens} )"
                 " {region.region}"),
                ("linalg.generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " init( {init_args.ssa_id_list} : {init_types.type_list_no_parens} )"
                 " {region.region} -> {out_type.type}")]


@dataclass
class LinalgIndexedGeneric(DialectOp):
    inargs: List[ast.SsaId]
    in_types: List[ast.Type]
    region: ast.Region
    outargs: Optional[List[ast.SsaId]] = None
    out_types: Optional[List[ast.Type]] = None
    init_args: Optional[List[ast.SsaId]] = None
    init_types: Optional[List[ast.Type]] = None
    out_type: Optional[ast.Type] = None
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.indexed_generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " outs( {outargs.ssa_id_list} : {out_types.type_list_no_parens} )"
                 " {region.region}"),
                ("linalg.indexed_generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " init( {init_args.ssa_id_list} : {init_types.type_list_no_parens} )"
                 " {region.region} -> {out_type.type}")]


@dataclass
class LinalgRange(DialectOp):
    min_id: ast.SsaId
    max_id: ast.SsaId
    step_id: ast.SsaId
    out_type: ast.Type
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.range {min_id.ssa_id} : {max_id.ssa_id} : {step_id.ssa_id}"
                 " {attr.attribute_value} : {out_type.type}"),
                ("linalg.range {min_id.ssa_id} : {max_id.ssa_id} : {step_id.ssa_id}"
                 " : {out_type.type}")]


@dataclass
class LinalgReshape(DialectOp):
    src_id: ast.SsaId
    src_type: ast.MemRefType
    result_type: ast.MemRefType
    reassociation: Optional[List[ast.AffineMap]] = None
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " {attr.attribute_value} "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_id}"
                 " [ ] "
                 " {attr.attribute_value} "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_id}"
                 " [ ] "
                 " : {src_type.memref_type} into {result_type.memref_type}")]


@dataclass
class LinalgSlice(DialectOp):
    view_id: ast.SsaId
    indexing_ids: List[ast.SsaId]
    view_type: ast.Type
    indexing_types: List[ast.Type]
    result_type: ast.Type

    _syntax_ = ("linalg.slice {view_id.ssa_id} [ {indexing_ids.ssa_id_list} ]"
                " : {view_type.type} , {indexing_types.type_list_no_parens} "
                " , {result_type.type}")


@dataclass
class TensorReshape(DialectOp):
    src_id: ast.SsaId
    src_type: ast.MemRefType
    result_type: ast.MemRefType
    reassociation: Optional[List[ast.AffineMap]] = None
    attr: Optional[ast.Attribute] = None

    _syntax_ = [("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " {attr.attribute_value} "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ ] "
                 " {attr.attribute_value} "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ ] "
                 " : {src_type.tensor_type} into {result_type.tensor_type}")]


@dataclass
class LinalgYield(DialectOp):
    operand_ids: List[ast.SsaId]
    operand_types: List[ast.Type]

    _syntax_ = ("linalg.yield {operand_ids.ssa_id_list}"
                " : {operand_types.type_list_no_parens}")


@dataclass
class LinalgMatmul(DialectOp):
    a_id: ast.SsaId
    b_id: ast.SsaId
    a_type: ast.Type
    b_type: ast.Type
    c_id: ast.SsaId
    c_type: ast.Type
    out_type: Optional[ast.Type] = None

    _syntax_ = [("linalg.matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_id} : {c_type.type} )"),
                ("linalg.matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " init( {c_id.ssa_id} : {c_type.type} )  -> {out_type.type}")]


@dataclass
class LinalgMatvec(DialectOp):
    a_id: ast.SsaId
    b_id: ast.SsaId
    a_type: ast.Type
    b_type: ast.Type
    c_id: ast.SsaId
    c_type: ast.Type

    _syntax_ = [("linalg.matvec"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_id} : {c_type.type} )")]


# Inspect current module to get all classes defined above
linalg = Dialect("linalg", ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
