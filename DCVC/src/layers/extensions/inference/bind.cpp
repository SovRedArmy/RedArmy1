// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// pybind11 绑定入口：导出 DMCHTSProxy / DMCHTLProxy / DMCLDProxy / DMCIProxy
// 为 Python 模块 inference_extensions_cuda。Python 侧 DMC.compress/decompress
// 与 DMCI.compress/decompress 即转入这里的 C++/CUDA 实现。

#include <torch/extension.h>

#include "dmc_htl_proxy.h"
#include "dmc_hts_proxy.h"
#include "dmc_ld_proxy.h"
#include "dmci_proxy.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<DMCIProxy>(m, "DMCIProxy")
        .def(py::init<>())
        .def("set_param", &DMCIProxy::set_param)
        .def("compress", &DMCIProxy::compress)
        .def("decompress", &DMCIProxy::decompress);

    py::class_<DMCHTLProxy>(m, "DMCHTLProxy")
        .def(py::init<>())
        .def("set_param", &DMCHTLProxy::set_param)
        .def("add_ref_feature_from_frame", &DMCHTLProxy::add_ref_feature_from_frame)
        .def("compress", &DMCHTLProxy::compress)
        .def("decompress", &DMCHTLProxy::decompress);

    py::class_<DMCHTSProxy>(m, "DMCHTSProxy")
        .def(py::init<>())
        .def("set_param", &DMCHTSProxy::set_param)
        .def("add_ref_feature_from_frame", &DMCHTSProxy::add_ref_feature_from_frame)
        .def("compress", &DMCHTSProxy::compress)
        .def("decompress", &DMCHTSProxy::decompress);

    py::class_<DMCLDProxy>(m, "DMCLDProxy")
        .def(py::init<>())
        .def("set_param", &DMCLDProxy::set_param)
        .def("add_ref_feature_from_frame", &DMCLDProxy::add_ref_feature_from_frame)
        .def("compress", &DMCLDProxy::compress)
        .def("decompress", &DMCLDProxy::decompress);
}
