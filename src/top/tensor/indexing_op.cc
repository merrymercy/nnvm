/*!
 *  Copyright (c) 2018 by Contributors
 * \file indexing_op.cc
 * \brief indexing operator registration
 */

#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include "../indexing_op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

DMLC_REGISTER_PARAMETER(EmbeddingParam);

NNVM_REGISTER_OP(embedding)
    .describe(R"(Maps integer indices to vector representations (embeddings).
)" NNVM_ADD_FILELINE)
.add_argument("data", "1D Tensor", "The input array to the embedding operator.")
.add_argument("weight", "2D Tensor", "The embedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__())
.set_attr_parser(ParamParser<EmbeddingParam>)
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
   return std::vector<std::string>{"data", "weight"};
})
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<EmbeddingParam>)
.set_attr<FInferShape>("FInferShape", EmbeddingOpShape)
.set_attr<FInferType>("FInferType", EmbeddingOpType)
.set_num_inputs(2)
.set_num_outputs(1)
.set_support_level(4);

}  // namespace top
}  // namespace nnvm
