#pragma once

#include <vector>
#include <algorithm>
#include <cuda.h>

#include "gpu/mblas/matrix.h"
#include "gpu/mblas/vector.h"
#include "nth_element_kernels.h"

namespace amunmt {
namespace GPU {


class NthElement {
  public:
    NthElement() = delete;
    NthElement(const NthElement &copy) = delete;
    NthElement(unsigned maxBeamSize, unsigned maxBatchSize);
    virtual ~NthElement();

    // standard nth_element
    void getNBestList(const std::vector<unsigned>& beamSizes,
                      mblas::Matrix& Probs,
                      std::vector<float>& outCosts,
                      std::vector<unsigned>& outKeys,
                      const bool isFirst=false);

    void GetPairs(unsigned number,
                  std::vector<unsigned>& outKeys,
                  std::vector<float>& outValues);

    void getValueByKey(std::vector<float>& out, const mblas::Matrix &d_in) const;

  private:
    const unsigned BLOCK_SIZE = 512;

    mblas::Vector<NthOut> d_out;

    mblas::Vector<NthOut> d_res;
    std::vector<NthOut> h_res;

    mblas::Vector<float> d_breakdown;
    mblas::Vector<unsigned> d_batchPosition;
    mblas::Vector<unsigned> d_cumBeamSizes;

    unsigned maxBeamSize_, maxBatchSize_;

    void getNBestList(mblas::Matrix &probs,
                      const std::vector<unsigned>& batchFirstElementIdxs,
                      const std::vector<unsigned>& cummulatedBeamSizes);


};

}  // namespace GPU
}
