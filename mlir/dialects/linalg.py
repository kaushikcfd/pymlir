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
from mlir.dialect import Dialect, DialectOp, is_op


class LinalgBatchMatmul(DialectOp):
    _syntax_ = [("linalg.batch_matmul"
                 " ins( {a_id.ssa_use} , {b_id.ssa_use} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_use} : {c_type.type} )"),
                ("linalg.batch_matmul"
                 " ins( {a_id.ssa_use} , {b_id.ssa_use} : {a_type.type} , {b_type.type} )"
                 " init( {init_id.ssa_use} : {init_type.type} ) -> {out_type.type}")]


class LinalgConvW(DialectOp):
    _syntax_ = [("linalg.conv_1d"
                 " ins( {in_id.ssa_use} , {filter_id.ssa_use} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_use} : {out_type.type} )")]


class LinalgConvHW(DialectOp):
    _syntax_ = [("linalg.conv_2d"
                 " ins( {in_id.ssa_use} , {filter_id.ssa_use} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_use} : {out_type.type} )")]


class LinalgConvDHW(DialectOp):
    _syntax_ = [("linalg.conv_3d"
                 " ins( {in_id.ssa_use} , {filter_id.ssa_use} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_use} : {out_type.type} )")]


class LinalgConv(DialectOp):
    _syntax_ = [("linalg.conv( {in_id.ssa_use} , {filter_id.ssa_use} , {out_id.ssa_use} ) "
                "{attr.attribute_value} : {in_type.type} , {filter_type.type} , {out_type.type}"),
                ("linalg.conv( {in_id.ssa_use} , {filter_id.ssa_use} , {out_id.ssa_use} ) "
                " : {in_type.type} , {filter_type.type} , {out_type.type}")]


class LinalgCopy(DialectOp):
    _syntax_ = [("linalg.copy( {a_id.ssa_use} , {b_id.ssa_use} ) "
                "{attr.attribute_value} : {a_type.type} , {b_type.type}"),
                ("linalg.copy( {a_id.ssa_use} , {b_id.ssa_use} ) "
                " : {a_type.type} , {b_type.type}")]


class LinalgDot(DialectOp):
    _syntax_ = [("linalg.dot"
                 " ins( {in_a_id.ssa_use} , {in_b_id.ssa_use} : {in_a_type.type} , {in_b_type.type} )"
                 " outs( {out_id.ssa_use} : {out_type.type} )")]


class LinalgFill(DialectOp):
    _syntax_ = [("linalg.fill( {output_id.ssa_use} , {value_id.ssa_use} ) "
                "{attr.attribute_value} : {output_type.type} , {value_type.type}"),
                ("linalg.fill( {output_id.ssa_use} , {value_id.ssa_use} ) "
                " : {output_type.type} , {value_type.type}")]


class LinalgGeneric(DialectOp):
    _syntax_ = [("linalg.generic {attr.attribute_value} "
                 " ins( {inargs.ssa_use_list} : {in_types.type_list_no_parens} )"
                 " outs( {outargs.ssa_use_list} : {out_types.type_list_no_parens} )"
                 " {body.region}"),
                ("linalg.generic {attr.attribute_value} "
                 " ins( {inargs.ssa_use_list} : {in_types.type_list_no_parens} )"
                 " init( {init_args.ssa_use_list} : {init_types.type_list_no_parens} )"
                 " {body.region} -> {out_type.type}")]


class LinalgIndexedGeneric(DialectOp):
    _syntax_ = [("linalg.indexed_generic {attr.attribute_value} "
                 " ins( {inargs.ssa_use_list} : {in_types.type_list_no_parens} )"
                 " outs( {outargs.ssa_use_list} : {out_types.type_list_no_parens} )"
                 " {body.region}"),
                ("linalg.indexed_generic {attr.attribute_value} "
                 " ins( {inargs.ssa_use_list} : {in_types.type_list_no_parens} )"
                 " init( {init_args.ssa_use_list} : {init_types.type_list_no_parens} )"
                 " {body.region} -> {out_type.type}")]


class LinalgRange(DialectOp):
    _syntax_ = [("linalg.range {min_id.ssa_use} : {max_id.ssa_use} : {step_id.ssa_use}"
                 " {attr.attribute_value} : {out_type.type}"),
                ("linalg.range {min_id.ssa_use} : {max_id.ssa_use} : {step_id.ssa_use}"
                 " : {out_type.type}")]


class LinalgReshape(DialectOp):
    _syntax_ = [("linalg.reshape {src_id.ssa_use}"
                 " [ {reassociation.affine_map_list} ] "
                 " {attr.attribute_value} "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_use}"
                 " [ ] "
                 " {attr.attribute_value} "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_use}"
                 " [ {reassociation.affine_map_list} ] "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_use}"
                 " [ ] "
                 " : {src_type.memref_type} into {result_type.memref_type}")]


class LinalgSlice(DialectOp):
    _syntax_ = ("linalg.slice {view_id.ssa_use} [ {indexing_ids.ssa_use_list} ]"
                " : {view_type.type} , {indexing_types.type_list_no_parens} "
                " , {result_type.type}")


class TensorReshape(DialectOp):
    _syntax_ = [("linalg.tensor_reshape {src_id.ssa_use}"
                 " [ {reassociation.affine_map_list} ] "
                 " {attr.attribute_value} "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_use}"
                 " [ ] "
                 " {attr.attribute_value} "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_use}"
                 " [ {reassociation.affine_map_list} ] "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_use}"
                 " [ ] "
                 " : {src_type.tensor_type} into {result_type.tensor_type}")]


class LinalgYield(DialectOp):
    _syntax_ = ("linalg.yield {operand_ids.ssa_use_list}"
                " : {operand_types.type_list_no_parens}")


class LinalgMatmul(DialectOp):
    _syntax_ = [("linalg.matmul"
                 " ins( {a_id.ssa_use} , {b_id.ssa_use} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_use} : {c_type.type} )"),
                ("linalg.matmul"
                 " ins( {a_id.ssa_use} , {b_id.ssa_use} : {a_type.type} , {b_type.type} )"
                 " init( {init_id.ssa_use} : {init_type.type} )  -> {out_type.type}")]


class LinalgMatvec(DialectOp):
    _syntax_ = [("linalg.matvec"
                 " ins( {a_id.ssa_use} , {b_id.ssa_use} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_use} : {c_type.type} )")]


# Inspect current module to get all classes defined above
linalg = Dialect("linalg", ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
