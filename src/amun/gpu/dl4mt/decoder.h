#pragma once

#include "gpu/mblas/matrix_functions.h"
#include "model.h"
#include "gru.h"
#include "gpu/types-gpu.h"
#include "common/god.h"
#include "../decoder/beam_size_gpu.h"

namespace amunmt {
namespace GPU {

class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Rows, const std::vector<size_t>& ids) {
          using namespace mblas;
          HostVector<uint> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_->dim(0))
              id = 1;
          indices_.resize(tids.size());

          mblas::copy(thrust::raw_pointer_cast(tids.data()),
              tids.size(),
              thrust::raw_pointer_cast(indices_.data()),
              cudaMemcpyHostToDevice);

          Assemble(Rows, *w_.E_, indices_);
        }

        size_t GetCols() const
        { return w_.E_->dim(1); }

        size_t GetRows() const {
          return w_.E_->dim(0);
        }

      private:
        const Weights& w_;
        DeviceVector<uint> indices_;

        Embeddings(const Embeddings&) = delete;
    };

    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel)
        , gru_(gruModel)
        {}

        void InitializeState(mblas::Matrix& State,
                             const mblas::Matrix& sourceContext,
                             const size_t batchSize,
                             const mblas::IMatrix &sentenceLengths) const
        {
          using namespace mblas;

          //std::cerr << "State1=" << State.Debug(1) << std::endl;
          mblas::Matrix Temp2;
          Temp2.NewSize(batchSize, sourceContext.dim(1), 1, 1);
          //std::cerr << "2Temp2_=" << Temp2.Debug(1) << std::endl;

          //std::cerr << "sourceContext=" << sourceContext.Debug(1) << std::endl;
          //std::cerr << "mapping=" << Debug(mapping, 2) << std::endl;
          Mean(Temp2, sourceContext, sentenceLengths);

          //std::cerr << "1State=" << State.Debug(1) << std::endl;
          //std::cerr << "3Temp2_=" << Temp2.Debug(1) << std::endl;
          //std::cerr << "w_.Wi_=" << w_.Wi_->Debug(1) << std::endl;
          Prod(State, Temp2, *w_.Wi_);

          //std::cerr << "2State=" << State.Debug(1) << std::endl;
          //State.ReduceDimensions();

          if (w_.Gamma_->size()) {
            Normalization(State, State, *w_.Gamma_, *w_.Bi_, 1e-9);
            Element(Tanh(_1), State);
          } else {
            BroadcastVec(Tanh(_1 + _2), State, *w_.Bi_);
          }
          //std::cerr << "State2=" << State.Debug(1) << std::endl;
          //std::cerr << "\n";
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) const
        {
          gru_.GetNextState(NextState, State, Context);
        }

      private:
        const Weights1& w_;
        const GRU<Weights2> gru_;

        RNNHidden(const RNNHidden&) = delete;
    };

    template <class Weights>
    class RNNFinal {
      public:
        RNNFinal(const Weights& model)
        : gru_(model) {}

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) const
        {
          gru_.GetNextState(NextState, State, Context);
        }

      private:
        const GRU<Weights> gru_;

        RNNFinal(const RNNFinal&) = delete;
    };

    template <class Weights>
    class Alignment {
      public:
        Alignment(const God &god, const Weights& model)
          : w_(model)
        {}

        void Init(mblas::Matrix &SCU, const mblas::Matrix& sourceContext) const
        {
          using namespace mblas;

          Prod(/*h_[0],*/ SCU, sourceContext, *w_.U_);
          //std::cerr << "SCU_=" << SCU_.Debug(1) << std::endl;

          if (w_.Gamma_1_->size()) {
            Normalization(SCU, SCU, *w_.Gamma_1_, *w_.B_, 1e-9);
          }
        }

        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     mblas::Matrix &attention,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& sourceContext,
                                     const mblas::IMatrix &sentenceLengths,
                                     const mblas::Matrix& SCU,
                                     const BeamSizeGPU& beamSizes) const
        {
          // mapping = 1/0 whether each position, in each sentence in the batch is actually a valid word
          // batchMapping = which sentence is each element in the batch. eg 0 0 1 2 2 2 = first 2 belongs to sent0, 3rd is sent1, 4th and 5th is sent2
          // dBatchMapping = fixed length (batch*beam) version of dBatchMapping_

          using namespace mblas;

          mblas::Matrix Temp1;
          mblas::Matrix Temp2;

          size_t maxLength = sourceContext.dim(0);
          size_t batchSize = sourceContext.dim(3);
          //std::cerr << "batchSize=" << batchSize << std::endl;
          //std::cerr << "HiddenState=" << HiddenState.Debug(0) << std::endl;

          HostVector<uint> batchMapping(HiddenState.dim(0));
          size_t k = 0;
          for (size_t i = 0; i < beamSizes.size(); ++i) {
            for (size_t j = 0; j < beamSizes.Get(i).size; ++j) {
              batchMapping[k++] = i;
            }
          }
          //std::cerr << "batchMapping=" << mblas::Debug(batchMapping, 2) << std::endl;

          DeviceVector<uint> dBatchMapping(batchMapping.size());
          mblas::copy(thrust::raw_pointer_cast(batchMapping.data()),
              batchMapping.size(),
              thrust::raw_pointer_cast(dBatchMapping.data()),
              cudaMemcpyHostToDevice);

          //std::cerr << "sourceContext=" << sourceContext.Debug(0) << std::endl;
          //std::cerr << "sentenceLengths=" << sentenceLengths.Debug(2) << std::endl;
          //std::cerr << "maxLength=" << maxLength << std::endl;

          Prod(/*h_[1],*/ Temp2, HiddenState, *w_.W_);
          //std::cerr << "1Temp2_=" << Temp2.Debug() << std::endl;

          if (w_.Gamma_2_->size()) {
            Normalization(Temp2, Temp2, *w_.Gamma_2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, Temp2, *w_.B_/*, s_[1]*/);
          }
          //std::cerr << "2Temp2_=" << Temp2.Debug() << std::endl;

          Copy(Temp1, SCU);
          //std::cerr << "1Temp1_=" << Temp1.Debug() << std::endl;

          Broadcast(Tanh(_1 + _2), Temp1, Temp2, dBatchMapping, maxLength);

          //std::cerr << "w_.V_=" << w_.V_->Debug(0) << std::endl;
          //std::cerr << "3Temp1_=" << Temp1.Debug(0) << std::endl;

          Prod(attention, *w_.V_, Temp1, false, true);

          mblas::Softmax(attention, dBatchMapping, sentenceLengths, batchSize);
          mblas::WeightedMean(AlignedSourceContext, attention, sourceContext, dBatchMapping);

          /*
          std::cerr << "AlignedSourceContext=" << AlignedSourceContext.Debug(0) << std::endl;
          std::cerr << "attention=" << attention.Debug(0) << std::endl;
          std::cerr << "HiddenState=" << HiddenState.Debug(0) << std::endl;
          std::cerr << "sourceContext=" << sourceContext.Debug(0) << std::endl;
          std::cerr << "sentenceLengths=" << sentenceLengths.Debug(0) << std::endl;
          std::cerr << "SCU=" << SCU.Debug(0) << std::endl;
          std::cerr << "beamSizes=" << beamSizes.Debug(0) << std::endl;
          */
        }

      private:
        const Weights& w_;

        Alignment(const Alignment&) = delete;
    };

    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          mblas::Transpose(TempW4, *w_.W4_);
          mblas::Transpose(TempB4, *w_.B4_);
        }

        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) const
        {
          using namespace mblas;

          mblas::Matrix T1;
          mblas::Matrix T2;
          mblas::Matrix T3;

          //BEGIN_TIMER("GetProbs.Prod");
          Prod(/*h_[0],*/ T1, State, *w_.W1_);
          //PAUSE_TIMER("GetProbs.Prod");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec");
          if (w_.Gamma_1_->size()) {
            Normalization(T1, T1, *w_.Gamma_1_, *w_.B1_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T1, *w_.B1_ /*,s_[0]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec");

          //BEGIN_TIMER("GetProbs.Prod2");
          Prod(/*h_[1],*/ T2, Embedding, *w_.W2_);
          //PAUSE_TIMER("GetProbs.Prod2");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec2");
          if (w_.Gamma_0_->size()) {
            Normalization(T2, T2, *w_.Gamma_0_, *w_.B2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T2, *w_.B2_ /*,s_[1]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec2");

          //BEGIN_TIMER("GetProbs.Prod3");
          Prod(/*h_[2],*/ T3, AlignedSourceContext, *w_.W3_);
          //PAUSE_TIMER("GetProbs.Prod3");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec3");
          if (w_.Gamma_2_->size()) {
            Normalization(T3, T3, *w_.Gamma_2_, *w_.B3_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T3, *w_.B3_ /*,s_[2]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec3");

          //BEGIN_TIMER("GetProbs.Element");
          Element(Tanh(_1 + _2 + _3), T1, T2, T3);
          //PAUSE_TIMER("GetProbs.Element");

          std::shared_ptr<mblas::Matrix> w4, b4;
          if(!filtered_) {
            w4 = w_.W4_;
            b4 = w_.B4_;
          } else {
            w4.reset(&FilteredW4_);
            b4.reset(&FilteredB4_);
          }

          BEGIN_TIMER("GetProbs.Prod4");
          Prod(Probs, T1, *w4);
          PAUSE_TIMER("GetProbs.Prod4");

          BEGIN_TIMER("GetProbs.BroadcastVec");
          BroadcastVec(_1 + _2, Probs, *b4);
          PAUSE_TIMER("GetProbs.BroadcastVec");

          BEGIN_TIMER("GetProbs.LogSoftMax");
          mblas::LogSoftmax(Probs);
          PAUSE_TIMER("GetProbs.LogSoftMax");
        }

        void Filter(const std::vector<size_t>& ids) {
          filtered_ = true;
          using namespace mblas;

          Assemble(FilteredW4_, TempW4, ids);
          Assemble(FilteredB4_, TempB4, ids);

          Transpose(FilteredW4_);
          Transpose(FilteredB4_);
        }

      private:
        const Weights& w_;

        bool filtered_;
        mutable mblas::Matrix FilteredW4_;
        mutable mblas::Matrix FilteredB4_;

        mblas::Matrix TempW4;
        mblas::Matrix TempB4;

        Softmax(const Softmax&) = delete;
    };

  public:
    Decoder(const God &god, const Weights& model)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, model.decGru1_),
      rnn2_(model.decGru2_),
      alignment_(god, model.decAlignment_),
      softmax_(model.decSoftmax_)
    {}

    void Decode(mblas::Matrix& NextState,
                mblas::Matrix &probs,
                mblas::Matrix &attention,
                const mblas::Matrix& State,
                const mblas::Matrix& Embeddings,
                const mblas::Matrix& SCU,
                const BeamSizeGPU& beamSizes) const
    {
      //BEGIN_TIMER("Decoder.Decode");

      mblas::Matrix HiddenState;
      mblas::Matrix AlignedSourceContext;

      BEGIN_TIMER("Decode.GetHiddenState");
      //std::cerr << "State=" << State.Debug(1) << std::endl;
      //std::cerr << "Embeddings=" << Embeddings.Debug(1) << std::endl;
      GetHiddenState(HiddenState, State, Embeddings);
      //HiddenState.ReduceDimensions();
      //std::cerr << "HiddenState=" << HiddenState.Debug(1) << std::endl;
      PAUSE_TIMER("Decode.GetHiddenState");

      BEGIN_TIMER("Decode.GetAlignedSourceContext");
      GetAlignedSourceContext(AlignedSourceContext,
                              attention,
                              HiddenState,
                              beamSizes.GetSourceContext(),
                              beamSizes.GetSentenceLengths(),
                              SCU,
                              beamSizes);
      //std::cerr << "AlignedSourceContext=" << AlignedSourceContext.Debug(1) << std::endl;
      PAUSE_TIMER("Decode.GetAlignedSourceContext");

      BEGIN_TIMER("Decode.GetNextState");
      GetNextState(NextState, HiddenState, AlignedSourceContext);
      //std::cerr << "NextState=" << NextState.Debug(1) << std::endl;
      PAUSE_TIMER("Decode.GetNextState");

      BEGIN_TIMER("Decode.GetProbs");
      GetProbs(probs, NextState, Embeddings, AlignedSourceContext);
      //std::cerr << "Probs_=" << Probs_.Debug(1) << std::endl;
      PAUSE_TIMER("Decode.GetProbs");

      //PAUSE_TIMER("Decoder.Decode");
    }

    void EmptyState(mblas::Matrix& State,
                    mblas::Matrix &SCU,
                    const mblas::Matrix &sourceContext,
                    const mblas::IMatrix &sourceLengths,
                    size_t batchSize) const
    {
      rnn1_.InitializeState(State,
                            sourceContext,
                            batchSize,
                            sourceLengths);

      alignment_.Init(SCU, sourceContext);
    }

    void EmptyEmbedding(mblas::Matrix& Embedding, size_t batchSize = 1) const
    {
      Embedding.NewSize(batchSize, embeddings_.GetCols());
      mblas::Zero(Embedding);
    }

    void Lookup(mblas::Matrix& Embedding,
                const std::vector<size_t>& w) {
      embeddings_.Lookup(Embedding, w);
    }

    void Filter(const std::vector<size_t>& ids) {
      softmax_.Filter(ids);
    }

    size_t GetVocabSize() const {
      return embeddings_.GetRows();
    }

  private:

    void GetHiddenState(mblas::Matrix& HiddenState,
                        const mblas::Matrix& PrevState,
                        const mblas::Matrix& Embedding) const
    {
      rnn1_.GetNextState(HiddenState, PrevState, Embedding);
    }

    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                  mblas::Matrix &probs,
                                  const mblas::Matrix& HiddenState,
                                  const mblas::Matrix& sourceContext,
                                  const mblas::IMatrix &sentenceLengths,
                                  const mblas::Matrix& SCU,
                                  const BeamSizeGPU& beamSizes) const
    {
      alignment_.GetAlignedSourceContext(AlignedSourceContext,
                                          probs,
                                          HiddenState,
                                          sourceContext,
                                          sentenceLengths,
                                          SCU,
                                          beamSizes);
    }

    void GetNextState(mblas::Matrix& State,
                      const mblas::Matrix& HiddenState,
                      const mblas::Matrix& AlignedSourceContext) const
    {
      rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
    }


    void GetProbs(mblas::Matrix &Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) const
    {
      softmax_.GetProbs(Probs, State, Embedding, AlignedSourceContext);
    }

  private:

    Embeddings<Weights::DecEmbeddings> embeddings_;
    RNNHidden<Weights::DecInit, Weights::DecGRU1> rnn1_;
    RNNFinal<Weights::DecGRU2> rnn2_;
    Alignment<Weights::DecAlignment> alignment_;
    Softmax<Weights::DecSoftmax> softmax_;

    Decoder(const Decoder&) = delete;
};

}
}

