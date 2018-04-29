/*!
 *  Copyright (c) 2018 by Contributors
 * \file indexing_op_common.h
 * \brief Common indexing operator utilities
 */
#ifndef NNVM_TOP_INDEXING_OP_COMMON_H_
#define NNVM_TOP_INDEXING_OP_COMMON_H_

#include <string>
#include <vector>
#include <utility>
#include <nnvm/node.h>
#include "./op_common.h"
#include "termio.h"

namespace nnvm {
namespace top {

inline bool EmbeddingOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  const EmbeddingParam& param = nnvm::get<EmbeddingParam>(attrs.parsed);
  const TShape &dshape = (*in_shape)[0];
  const TShape &wshape = (*in_shape)[1];

  if (dshape.ndim() ==  0) return false;

  CHECK_EQ(wshape.ndim(), 2U);
  CHECK_EQ(wshape[0], param.input_dim);
  CHECK_EQ(wshape[1], param.output_dim);

  out_shape->clear();
  TShape oshape(dshape.ndim()+1);
  for (size_t i = 0; i < dshape.ndim(); ++i) {
    oshape[i] = dshape[i];
  }
  oshape[dshape.ndim()] = param.output_dim;

  out_shape->push_back(oshape);
  return true;
}

inline bool EmbeddingOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type,
                            std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2U);
  CHECK_GE(out_type->size(), 1U);
  int itype = (*in_type)[0];
  CHECK_NE(itype, -1) << "First input must have specified type";
  int dtype_in = (*in_type)[1];
  int dtype_out = (*out_type)[0];
  int dtype;
  if (dtype_in != -1 && dtype_out != -1) {
    // Both types defined, make sure they are the same
    CHECK_EQ(dtype_in, dtype_out) << "Input and output weights must have same type";
    dtype = dtype_in;
  } else if (dtype_in != -1 || dtype_out != -1) {
    // One of the types defined, choose the one that was defined
    dtype = (dtype_in != -1) ? dtype_in : dtype_out;
  }
  if ((*in_type)[1] == -1) (*in_type)[1] = dtype;
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}

}  // namespace top
}  // namespace nnvm
#endif  // NNVM_TOP_INDEXING_OP_COMMON_H_
