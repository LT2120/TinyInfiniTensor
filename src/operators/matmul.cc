#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        
        auto A = inputs[0]->getDims();
        auto B = inputs[1]->getDims();

        Shape out(std::max(A.size(),B.size()));//operator_utils broadcast
        auto i = A.rbegin();
        auto j = B.rbegin();
        auto k = out.rbegin();
        size_t a,b;
        while( k!=out.rend()){
            if(i==A.rend()){
                a = 1;
            }else{
                a = *i;
            }
            if(j==B.rend()){
                b = 1;
            }else{
                b = *j;
            }
            *k = std::max(a,b);
            if(i!=A.rend())++i;
            if(j!=B.rend())++j;
            ++k;
        }

        i = A.rbegin();
        j = B.rbegin();
        k = out.rbegin();

        if(transB)++j;
        *k = *j;
        ++k;
        if(!transA)++i;
        *k = *i;

        return {{out}};
    }

} // namespace infini