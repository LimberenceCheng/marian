// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"
#include "common/search.h"
#include "common/histories.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/decoder/enc_out_gpu.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"
#include "gpu/decoder/beam_size_gpu.h"

using namespace std;

namespace amunmt {
namespace GPU {

std::unordered_map<std::string, boost::timer::cpu_timer> timers;

///////////////////////////////////////////////////////////////////////////////
EncoderDecoder::EncoderDecoder(
        const God &god,
        const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model,
        const Search &search)
: Scorer(god, name, config, tab, search),
  model_(model),
  encoder_(new Encoder(model_)),
  decoder_(new Decoder(god, model_)),
  encDecBuffer_(god.Get<size_t>("encoder-buffer-size"))

{
  BEGIN_TIMER("EncoderDecoder");

  std::thread *thread = new std::thread( [&]{ DecodeAsync(god); });
  decThread_.reset(thread);
}

EncoderDecoder::~EncoderDecoder()
{
  decThread_->join();
  PAUSE_TIMER("EncoderDecoder");

  if (timers.size()) {
    boost::timer::nanosecond_type encDecWall = timers["EncoderDecoder"].elapsed().wall;

    cerr << "timers:" << endl;
    for (auto iter = timers.begin(); iter != timers.end(); ++iter) {
      const boost::timer::cpu_timer &timer = iter->second;
      boost::timer::cpu_times t = timer.elapsed();
      boost::timer::nanosecond_type wallTime = t.wall;

      int percent = (float) wallTime / (float) encDecWall * 100.0f;

      cerr << iter->first << " ";

      for (int i = 0; i < ((int)35 - (int)iter->first.size()); ++i) {
        cerr << " ";
      }

      cerr << timer.format(2, "%w") << " (" << percent << ")" << endl;
    }
  }

}

State* EncoderDecoder::NewState() const {
  return new EDState();
}

void EncoderDecoder::Encode(const SentencesPtr source) {
  BEGIN_TIMER("Encode");

  EncOutPtr encOut(new EncOutGPU(source));

  if (source->size()) {
    //std::unique_lock<std::mutex> locker(mu);

    BEGIN_TIMER("Encode.Encode");
    encoder_->Encode(god_, *source, tab_, encOut);
    PAUSE_TIMER("Encode.Encode");

    mblas::Matrix &bufStates = encOut->GetStates<mblas::Matrix>();
    mblas::Matrix &bufEmbeddings = encOut->GetEmbeddings<mblas::Matrix>();

    const mblas::Matrix &sourceContext = encOut->GetSourceContext<mblas::Matrix>();
    const mblas::IMatrix &sourceLengths = encOut->GetSentenceLengths<mblas::IMatrix>();
    size_t batchSize = encOut->GetSentences().size();

    mblas::Matrix &SCU = encOut->GetSCU<mblas::Matrix>();

    //cerr << "1Encode:sourceContext=" << sourceContext.Debug(0) << endl;
    //cerr << "1Encode:SCU=" << SCU.Debug(0) << endl;
    //cerr << "1Encode:sourceLengths=" << sourceLengths.Debug(0) << endl;

    BEGIN_TIMER("Encode.BeginSentenceState");
    BeginSentenceState(bufStates,
                      bufEmbeddings,
                      SCU,
                      sourceContext,
                      sourceLengths,
                      batchSize);
    PAUSE_TIMER("Encode.BeginSentenceState");

    //cerr << "2Encode:sourceContext=" << sourceContext.Debug(0) << endl;
    //cerr << "2Encode:SCU=" << SCU.Debug(0) << endl;
    //cerr << "2Encode:sourceLengths=" << sourceLengths.Debug(0) << endl;
  }

  PAUSE_TIMER("Encode");

  BEGIN_TIMER("encDecBuffer_.Add");
  encDecBuffer_.Add(encOut);
  PAUSE_TIMER("encDecBuffer_.Add");
}

void EncoderDecoder::BeginSentenceState(mblas::Matrix &states,
                                        mblas::Matrix &embeddings,
                                        mblas::Matrix &SCU,
                                        const mblas::Matrix &sourceContext,
                                        const mblas::IMatrix &sourceLengths,
                                        size_t batchSize) const
{
  decoder_->EmptyState(states,
                      SCU,
                      sourceContext,
                      sourceLengths,
                      batchSize);

  decoder_->EmptyEmbedding(embeddings, batchSize);
}

void EncoderDecoder::DecodeAsync(const God &god)
{
  //cerr << "BeginSentenceState encOut->sourceContext_=" << encOut->sourceContext_.Debug(0) << endl;
  try {
    DecodeAsyncInternal(god);
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "CUDA error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "Bad memory allocation during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::runtime_error &e)
  {
    std::cerr << "Runtime error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(...)
  {
    std::cerr << "Some other kind of error during some_function" << std::endl;
    abort();
  }
}

void EncoderDecoder::DecodeAsyncInternal(const God &god)
{
  boost::timer::cpu_timer timer;

  BEGIN_TIMER("DecodeAsyncInternal.Total");
  //BEGIN_TIMER("DecodeAsyncInternal.Init");

  uint maxBeamSize = god.Get<uint>("beam-size");
  uint miniBatch = god.Get<uint>("mini-batch");
  size_t decodingBatch = god.Get<size_t>("decoding-mini-batch");
  if (decodingBatch) {
    miniBatch = decodingBatch;
  }

  EDState state;

  Hypotheses prevHyps;
  Histories histories(new BeamSizeGPU(), search_.NormalizeScore());
  mblas::Matrix SCU;
  size_t decoderStep = 0;

  // init batch
  // read in next batch
  EncOutPtr encOut = encDecBuffer_.Get();
  assert(encOut);

  const Sentences &sentences = encOut->GetSentences();
  if (sentences.size() == 0) {
    return;
  }

  timer.start();

  // init states & histories/beams
  const mblas::Matrix &bufStates = encOut->GetStates<mblas::Matrix>();
  const mblas::Matrix &bufEmbeddings = encOut->GetEmbeddings<mblas::Matrix>();
  const mblas::Matrix &bufSCU = encOut->GetSCU<mblas::Matrix>();

  mblas::Matrix &states = state.GetStates();
  mblas::Matrix &embeddings = state.GetEmbeddings();
  states.Copy(bufStates);
  embeddings.Copy(bufEmbeddings);
  SCU.Copy(bufSCU);

  histories.Init(maxBeamSize, encOut);
  prevHyps = histories.GetFirstHyps();

  //PAUSE_TIMER("DecodeAsyncInternal.Init");

  // MAIN LOOP
  while (histories.size()) {
    // decode
    boost::timer::cpu_timer timerStep;

    mblas::Matrix nextStateMatrix;
    mblas::Matrix attention;
    mblas::Matrix probs;

    const BeamSizeGPU &bsGPU = static_cast<const BeamSizeGPU&>(histories.GetBeamSizes());

    //std::unique_lock<std::mutex> locker(mu);

    BEGIN_TIMER("DecodeAsyncInternal.Decode");
    decoder_->Decode(nextStateMatrix,
                    probs,
                    attention,
                    state.GetStates(),
                    state.GetEmbeddings(),
                    SCU,
                    bsGPU);
    PAUSE_TIMER("DecodeAsyncInternal.Decode");

    // beams
    histories.SetNewBeamSize(search_.MaxBeamSize());

    BEGIN_TIMER("DecodeAsyncInternal.CalcBeam");
    Beams beams;
    search_.BestHyps()->CalcBeam(prevHyps, probs, attention, *this, search_.FilterIndices(), beams, histories.GetBeamSizes());
    PAUSE_TIMER("DecodeAsyncInternal.CalcBeam");

    //BEGIN_TIMER("DecodeAsyncInternal.AddAndOutput");
    std::pair<Hypotheses, std::vector<uint> > histOut = histories.AddAndOutput(god, beams);
    //PAUSE_TIMER("DecodeAsyncInternal.AddAndOutput");
    Hypotheses &survivors = histOut.first;
    const std::vector<uint> &completed = histOut.second;

    //BEGIN_TIMER("DecodeAsyncInternal.AssembleBeamState");
    AssembleBeamState(nextStateMatrix, survivors, state);
    //PAUSE_TIMER("DecodeAsyncInternal.AssembleBeamState");

    histories.SetFirst(false);

    size_t numNew;
    //numNew = completed.size();
    numNew = miniBatch - histories.size();
    //numNew = (maxBeamSize * miniBatch - survivors.size()) / maxBeamSize;

    if (numNew) {
      BEGIN_TIMER("DecodeAsyncInternal.numNew");
      std::vector<EncOut::SentenceElement> newSentences;

      //BEGIN_TIMER("DecodeAsyncInternal.encDecBuffer_.Get");
      encDecBuffer_.Get(numNew, newSentences);
      //PAUSE_TIMER("DecodeAsyncInternal.encDecBuffer_.Get");

      BeamSizeGPU &bsGPU2 = static_cast<BeamSizeGPU&>(histories.GetBeamSizes());

      BEGIN_TIMER("DecodeAsyncInternal.ShrinkBatch");
      ShrinkBatch(completed,
                  histories.GetBeamSizes(),
                  bsGPU2.GetSourceContext(),
                  bsGPU2.GetSentenceLengths(),
                  SCU);
      PAUSE_TIMER("DecodeAsyncInternal.ShrinkBatch");

      BEGIN_TIMER("DecodeAsyncInternal.AddToBatch");
      AddToBatch(newSentences,
                histories.GetBeamSizes(),
                bsGPU2.GetSourceContext(),
                bsGPU2.GetSentenceLengths(),
                SCU,
                state.GetStates(),
                state.GetEmbeddings());
      PAUSE_TIMER("DecodeAsyncInternal.AddToBatch");

      //BEGIN_TIMER("DecodeAsyncInternal.AddHypos");
      AddHypos(newSentences, survivors, histories);
      //PAUSE_TIMER("DecodeAsyncInternal.AddHypos");

      PAUSE_TIMER("DecodeAsyncInternal.numNew");
    }

    prevHyps.swap(survivors);
    ++decoderStep;


    LOG(progress)->info("Step {} took {}, batch size={}, survivors={}, completed={}",
                        decoderStep,
                        timerStep.format(3, "%ws"),
                        histories.size(),
                        survivors.size(),
                        completed.size()
                      );

    /*
    cerr << "3 nextState=" << nextStateMatrix.Debug(1) << endl;
    cerr << "3 probs=" << probs.Debug(1) << endl;
    cerr << "3 attention=" << attention.Debug(2) << endl;

    cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
    cerr << "survivors=" << survivors.size() << endl;
    cerr << "beams=" << beams.size() << endl;
    cerr << "beamSizes5=" << histories.GetBeamSizes().Debug(2) << endl;
    cerr << "histories=" << histories.size() << endl;
    cerr << "3 state=" << state.Debug(1) << endl;
    cerr << "3SCU=" << SCU.Debug(1) << endl;
    cerr << "completed=" << Debug(completed, 2) << endl;
    cerr << "newSentences=" << newSentences.size() << endl;
    cerr << endl;
    */
  }

  PAUSE_TIMER("DecodeAsyncInternal.Total");
}


void EncoderDecoder::AssembleBeamState(const mblas::Matrix &nextStateMatrix,
                                const Hypotheses& hypos,
                                EDState& out) const
{
  if (hypos.size() == 0) {
    return;
  }

  std::vector<size_t> beamWords;
  std::vector<uint> beamStateIds;
  for (const HypothesisPtr &h : hypos) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }

  DeviceVector<uint> indices(beamStateIds.size());
  //HostVector<uint> tmp = beamStateIds;

  mblas::copy(beamStateIds.data(),
      beamStateIds.size(),
      thrust::raw_pointer_cast(indices.data()),
      cudaMemcpyHostToDevice);

  mblas::Assemble(out.GetStates(), nextStateMatrix, indices);

  decoder_->Lookup(out.GetEmbeddings(), beamWords);
}

BaseMatrix& EncoderDecoder::GetProbs() {
  assert(false);
  return *(new mblas::Matrix());
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<size_t>& filterIds) {
  decoder_->Filter(filterIds);
}

void EncoderDecoder::ShrinkBatch(const std::vector<uint> &completed,
                                BeamSize &beamSize,
                                mblas::Matrix &sourceContext,
                                mblas::IMatrix &sentenceLengths,
                                mblas::Matrix &SCU)
{
  //cerr << "completed=" << Debug(completed, 2) << endl;
  if (completed.size() == 0) {
    return;
  }

  // shrink beam
  size_t origBeamSize = beamSize.size();
  beamSize.DeleteEmpty(completed);
  size_t newBeamSize = beamSize.size();

  // old ind -> new ind
  std::vector<uint> newIndices(newBeamSize, 99999);
  uint newInd = 0;
  uint completedInd = 0;
  for (size_t origInd = 0; origInd < origBeamSize; ++origInd) {
    if (completedInd < completed.size() && completed[completedInd] == origInd) {
      ++completedInd;
    }
    else {
      newIndices[newInd] = origInd;
      ++newInd;
    }
  }

  // shrink matrices
  //cerr << "BEFORE sourceContext=" << sourceContext.Debug(0) << endl;
  //cerr << "BEFORE SCU=" << SCU.Debug(0) << endl;
  //cerr << "BEFORE sentenceLengths=" << sentenceLengths.Debug(0) << endl;

  size_t sizeShrink = completed.size();
  DeviceVector<uint> d_newIndices(newIndices);

  uint maxLength = beamSize.GetMaxLength();

  ShrinkMatrix(sourceContext, 3, sizeShrink, d_newIndices, 0, maxLength);
  ShrinkMatrix(SCU, 3, sizeShrink, d_newIndices, 0, maxLength);

  ShrinkMatrix(sentenceLengths, 0, sizeShrink, d_newIndices);

  //cerr << "AFTER sourceContext=" << sourceContext.Debug(0) << endl;
  //cerr << "AFTER SCU=" << SCU.Debug(0) << endl;
  //cerr << "AFTER sentenceLengths=" << sentenceLengths.Debug(0) << endl;
}

void EncoderDecoder::AddToBatch(const std::vector<EncOut::SentenceElement> &newSentences,
                BeamSize &beamSize,
                mblas::Matrix &sourceContext,
                mblas::IMatrix &sentenceLengths,
                mblas::Matrix &SCU,
                mblas::Matrix &states,
                mblas::Matrix &embeddings)
{
  size_t currBatchInd = beamSize.size();
  size_t currHypoInd = states.dim(0);

  beamSize.AddNewSentences(newSentences);

  /*
  cerr << "BEFORE sourceContext=" << sourceContext.Debug(0) << endl;
  cerr << "BEFORE SCU=" << SCU.Debug(0) << endl;
  cerr << "BEFORE sentenceLengths=" << sentenceLengths.Debug(0) << endl;
  cerr << "BEFORE states=" << states.Debug(0) << endl;
  cerr << "BEFORE embeddings=" << embeddings.Debug(0) << endl;
  */

  uint numNewSentences = newSentences.size();
  uint maxLength = beamSize.GetMaxLength();
  //cerr << "maxLength=" << maxLength << endl;

  EnlargeMatrix(sourceContext, 3, numNewSentences, 0, maxLength);
  EnlargeMatrix(SCU, 3, numNewSentences, 0, maxLength);

  EnlargeMatrix(sentenceLengths, 0, numNewSentences);
  EnlargeMatrix(states, 0, numNewSentences);
  EnlargeMatrix(embeddings, 0, numNewSentences);

  /*
  cerr << "AFTER sourceContext=" << sourceContext.Debug(0) << endl;
  cerr << "AFTER SCU=" << SCU.Debug(0) << endl;
  cerr << "AFTER sentenceLengths=" << sentenceLengths.Debug(0) << endl;
  cerr << "AFTER states=" << states.Debug(0) << endl;
  cerr << "AFTER embeddings=" << embeddings.Debug(0) << endl;
  */

  for (size_t i = 0; i < newSentences.size(); ++i) {
    const EncOut::SentenceElement &ele = newSentences[i];
    const EncOutPtr encOut = ele.encOut;
    size_t sentenceInd = ele.sentenceInd;

    const mblas::Matrix &origSourceContext = encOut->GetSourceContext<mblas::Matrix>();
    const mblas::IMatrix &origSentenceLengths = encOut->GetSentenceLengths<mblas::IMatrix>();
    const mblas::Matrix &origSCU = encOut->GetSCU<mblas::Matrix>();
    const mblas::Matrix &origStates = encOut->GetStates<mblas::Matrix>();
    const mblas::Matrix &origEmbeddings = encOut->GetEmbeddings<mblas::Matrix>();

    /*
    cerr << "sentenceInd=" << sentenceInd << endl;
    cerr << "currBatchInd=" << currBatchInd << endl;
    cerr << "currHypoInd=" << currHypoInd << endl;
    cerr << "origSourceContext=" << origSourceContext.Debug(0) << endl;
    cerr << "origSentenceLengths=" << origSentenceLengths.Debug(0) << endl;
    cerr << "origSCU=" << origSCU.Debug(0) << endl;
    cerr << "origStates=" << origStates.Debug(0) << endl;
    cerr << "origEmbeddings=" << origEmbeddings.Debug(0) << endl;
    */

    assert(currBatchInd < sourceContext.dim(3));
    mblas::CopyDimension<float>(3, currBatchInd, sentenceInd, sourceContext, origSourceContext);

    assert(currBatchInd < sentenceLengths.dim(0));
    mblas::CopyDimension<uint>(0, currBatchInd, sentenceInd, sentenceLengths, origSentenceLengths);

    assert(currBatchInd < SCU.dim(3));
    mblas::CopyDimension<float>(3, currBatchInd, sentenceInd, SCU, origSCU);

    assert(currBatchInd < states.dim(0));
    mblas::CopyDimension<float>(0, currHypoInd, sentenceInd, states, origStates);

    assert(currBatchInd < embeddings.dim(0));
    mblas::CopyDimension<float>(0, currHypoInd, sentenceInd, embeddings, origEmbeddings);

    ++currBatchInd;
    ++currHypoInd;
  }
}

void EncoderDecoder::AddHypos(const std::vector<EncOut::SentenceElement> &newSentences,
              Hypotheses &survivors, Histories &histories)
{
  for (size_t i = 0; i < newSentences.size(); ++i) {
    const EncOut::SentenceElement &ele = newSentences[i];
    const Sentence &sentence = ele.GetSentence();

    HistoryPtr history = histories.Add(sentence);
    HypothesisPtr hypo = history->GetFirstHyps();

    survivors.push_back(HypothesisPtr(hypo));

  }

}

}
}

