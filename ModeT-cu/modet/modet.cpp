#include <torch/extension.h>
#include "utils.h"

torch::Tensor modet_fw(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb_opt
){  
    // feats: (B, 8, C)要被插值的z正方体8个顶点,坐标(0,0,0)到(1,1,1), point: (B,3) 待插值点
    CHECK_INPUT(query);
    CHECK_INPUT(key);
    const int heads = query.size(1);
    auto rpb = rpb_opt.has_value() ? rpb_opt.value() : torch::zeros({heads, 3, 3, 3}, query.options());
    assert(rpb.size(1) == 3);
    CHECK_INPUT(rpb);

    return modet_fw_cu(query, key, rpb);
}

std::vector<torch::Tensor> modet_bw(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled
){
    CHECK_INPUT(d_attn);
    CHECK_INPUT(query);
    CHECK_INPUT(key);

    return modet_bw_cu(d_attn, query, key, biasEnabled);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("modet_fw", &modet_fw);
    m.def("modet_bw", &modet_bw);
}
