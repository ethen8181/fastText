#include "index.h"
#include <future>

namespace fasttext {

void indexAddItem(
    const float* vectorData,
    const size_t& numFeatures,
    const size_t& labelId,
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw) {
    // function to add a single element to the index,
    // brought outside the index class so we can use it in a parallel context,
    // i.e. adding multiple items to the index in parallel.
    const float* point = vectorData + (numFeatures * labelId);
    hnsw->addPoint((void*)point, labelId);
}


Index::Index(
    const int32_t dim,
    const int32_t maxElements,
    const int32_t M,
    const int32_t efConstruction,
    const int32_t randomSeed) :
    dim_(dim), maxElements_(maxElements), M_(M),
    efConstruction_(efConstruction), randomSeed_(randomSeed) {
    
    // usually for hnswlib, there's a spaceName parameter
    // that accepts either "l2", "ip", or "cosine" as the similarity
    // metric, then creates the appropriate SpaceInterface object
    // accordingly, but for this use-case, we'll always be using the "ip",
    // inner product, similarity metric
    space_ = std::make_shared<hnswlib::InnerProductSpace>(dim);
    hnsw_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
        space_.get(), maxElements, M, efConstruction, randomSeed);
}


Index::Index(std::istream& in) {
    load(in);
}


void Index::addItems(std::shared_ptr<const fasttext::DenseMatrix> inputs) {
    const float* vectorData = inputs->data();
    std::vector<std::future<void>> allFuturePredictions(inputs->rows());
    for (size_t i = 0; i < inputs->rows(); i++) {
        allFuturePredictions[i] = std::async(
            indexAddItem,
            vectorData, inputs->cols(), i, hnsw_
        );
    }

    for (size_t i = 0; i < inputs->rows(); i++) {
        allFuturePredictions[i].get();
    }
}


std::vector<std::pair<float, size_t>>
Index::knnQuery(const fasttext::Vector& hidden, const int32_t k) {
    std::priority_queue<std::pair<float, size_t>> results = hnsw_->searchKnn(
        (void*)hidden.data(), k);

    std::vector<std::pair<float, size_t>> predictions(k);
    for (size_t i = 0; i < k; i++) {
        predictions[i] = results.top();
        results.pop();
    }

    // the priority_queue from searchKnn returns the data
    // in ascending order of the distance, we reverse the
    // vector to make it in descending order
    std::reverse(predictions.begin(), predictions.end());
    return predictions;
}


void Index::setEf(size_t ef) {
    hnsw_->ef_ = ef;
}


void Index::save(std::ostream& out) const {
    out.write((char*)&(dim_), sizeof(int32_t));
    out.write((char*)&(maxElements_), sizeof(int32_t));
    out.write((char*)&(M_), sizeof(int32_t));
    out.write((char*)&(efConstruction_), sizeof(int32_t));
    out.write((char*)&(randomSeed_), sizeof(int32_t));

    // the original method only accepts a string (file name)
    // we modified it to directly accept a ostream to
    // save it along with the upstream fasttext object
    hnsw_->saveIndex(out);
}


void Index::load(std::istream& in) {
    in.read((char*)&(dim_), sizeof(int32_t));
    in.read((char*)&(maxElements_), sizeof(int32_t));
    in.read((char*)&(M_), sizeof(int32_t));
    in.read((char*)&(efConstruction_), sizeof(int32_t));
    in.read((char*)&(randomSeed_), sizeof(int32_t));

    space_ = std::make_shared<hnswlib::InnerProductSpace>(dim_);
    // the original implementation has a
    // parameter nmslib = false, doesn't seem to be used
    // by the underlying call, hence we removed that parameter

    // we modify the original implementation of loading the model to
    // accept an istream, so we can load it along with the rest of
    // the fasttext model

    // another modification is to remove the code to check whether
    // the index is corrupted, since that code assumes this index
    // is saved independently, whereas in this use case, we're saving
    // the index along with the fasttext object, hence it would always
    // report that the index is corrupted
    hnsw_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
        space_.get(), in, maxElements_);
}

}
