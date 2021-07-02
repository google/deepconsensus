package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(name = "ipython")

# Modules for use in Colab. Update this to include other modules in Colab.
py_library(
    name = "deepconsensus_py",
    srcs = [],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus/models:data_providers",
        "//deepconsensus/models:losses_and_metrics",
        "//deepconsensus/models:majority_vote_transforms",
        "//deepconsensus/models:model_configs",
        "//deepconsensus/models:networks",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:colab_utils",
        "//deepconsensus/utils:dc_constants",
    ],
)

filegroup(
    name = "testdata",
    testonly = True,
    srcs = glob(
        [
            "testdata/**/*",
            "larger_testdata/**/*",
        ],
    ),
)
