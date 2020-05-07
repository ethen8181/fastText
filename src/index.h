# pragma once

#include "../third_party/hnswlib/hnswlib.h"
#include "densematrix.h"
#include "vector.h"
#include <memory>

namespace fasttext {

void indexAddItem(
    const float* vectorData,
    const size_t& numFeatures,
    const size_t& labelId,
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw);


class Index {
protected:
    int32_t dim_;
    int32_t maxElements_;
    int32_t M_;
    int32_t efConstruction_;
    int32_t randomSeed_;
    std::shared_ptr<hnswlib::SpaceInterface<float>> space_;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;

public:
    Index(
        const int32_t dim,
        const int32_t maxElements,
        const int32_t M,
        const int32_t efConstruction,
        const int32_t randomSeed);

    explicit Index(std::istream& in);

    void addItems(std::shared_ptr<const fasttext::DenseMatrix> inputs);

    std::vector<std::pair<float, size_t>>
    knnQuery(const fasttext::Vector& hidden, const int32_t k);

    void setEf(size_t ef);

    void save(std::ostream&) const;
    void load(std::istream&);
};

}
