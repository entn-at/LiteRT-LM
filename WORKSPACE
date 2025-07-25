# buildifier: disable=load-on-top

workspace(name = "litert_lm")

# UPDATED = 2025-06-18
LITERT_REF = "5caf8a4319b9a6f2fb64fde2d79d1c70ffd89e1f"
LITERT_SHA256 = "f304067e30232c3d1af107f9bbe52d1a05f29f980c4f5116a918026908c3101d"
TENSORFLOW_REF = "ee8fbba3e18b9991fac0e7c01dea07462bac0edb"
TENSORFLOW_SHA256 = "8ba6928d66f3b6a7a17aeb9282d1a850c6982d4b673f364dafe6c35ad3ebe61d"

# buildifier: disable=load-on-top

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

# Java rules
http_archive(
    name = "rules_java",
    sha256 = "c73336802d0b4882e40770666ad055212df4ea62cfa6edf9cb0f9d29828a0934",
    url = "https://github.com/bazelbuild/rules_java/releases/download/5.3.5/rules_java-5.3.5.tar.gz",
)

# Rust (for HuggingFace Tokenizers)
http_archive(
    name = "rules_rust",
    sha256 = "53c1bac7ec48f7ce48c4c1c6aa006f27515add2aeb05725937224e6e00ec7cea",
    url = "https://github.com/bazelbuild/rules_rust/releases/download/0.61.0/rules_rust-0.61.0.tar.gz",
    patches = ["@//:PATCH.rules_rust"],
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

rust_register_toolchains(extra_target_triples = [
    # Explicitly add toolchains for mobile. Desktop platforms are supported by default.
    "aarch64-linux-android",
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
])

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")

crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crates_repository")

crates_repository(
    name = "crate_index",
    cargo_lockfile = "//:Cargo.lock",
    lockfile = "//:cargo-bazel-lock.json",
    manifests = [
        "//:Cargo.toml",
    ],
)

load("@crate_index//:defs.bzl", "crate_repositories")

crate_repositories()

# TensorFlow
http_archive(
    name = "org_tensorflow",
    sha256 = TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-" + TENSORFLOW_REF,
    url = "https://github.com/tensorflow/tensorflow/archive/" + TENSORFLOW_REF + ".tar.gz",
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.9": "@org_tensorflow//:requirements_lock_3_9.txt",
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@local_xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@local_xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@local_xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@local_xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    name = "maven",
    artifacts = [
        "androidx.lifecycle:lifecycle-common:2.8.7",
        "com.google.android.play:ai-delivery:0.1.1-alpha01",
        "com.google.guava:guava:33.4.6-android",
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.1",
        "org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.10.1",
        "org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.10.1",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
)

# Kotlin rules
http_archive(
    name = "rules_kotlin",
    sha256 = "e1448a56b2462407b2688dea86df5c375b36a0991bd478c2ddd94c97168125e2",
    url = "https://github.com/bazelbuild/rules_kotlin/releases/download/v2.1.3/rules_kotlin-v2.1.3.tar.gz",
)

load("@rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()  # if you want the default. Otherwise see custom kotlinc distribution below

load("@rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()  # to use the default toolchain, otherwise see toolchains below

# Same one downloaded by tensorflow, but refer contrib/minizip.
http_archive(
    name = "minizip",
    add_prefix = "minizip",
    build_file = "@//:BUILD.minizip",
    sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    strip_prefix = "zlib-1.3.1/contrib/minizip",
    url = "https://zlib.net/fossils/zlib-1.3.1.tar.gz",
)

http_archive(
    name = "sentencepiece",
    build_file = "@//:BUILD.sentencepiece",
    patch_cmds = [
        # Empty config.h seems enough.
        "touch config.h",
        # Replace third_party/absl/ with absl/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/absl/|#include \"absl/|g' *.h *.cc",
        # Replace third_party/darts_clone/ with include/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/darts_clone/|#include \"include/|g' *.h *.cc",
    ],
    patches = ["@//:PATCH.sentencepiece"],
    sha256 = "9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86",
    strip_prefix = "sentencepiece-0.2.0/src",
    url = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz",
)

http_archive(
    name = "darts_clone",
    build_file = "@//:BUILD.darts_clone",
    sha256 = "4a562824ec2fbb0ef7bd0058d9f73300173d20757b33bb69baa7e50349f65820",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    url = "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.tar.gz",
)

http_archive(
    name = "litert",
    patch_cmds = [
        # Replace @//third_party with @litert//third_party in files under third_party/.
        "sed -i -e 's|\"@//third_party/|\"@litert//third_party/|g' third_party/*/*",
    ],
    sha256 = LITERT_SHA256,
    strip_prefix = "LiteRT-" + LITERT_REF,
    url = "https://github.com/google-ai-edge/LiteRT/archive/" + LITERT_REF + ".tar.gz",
)

http_archive(
    name = "tokenizers_cpp",
    build_file = "@//:BUILD.tokenizers_cpp",
    sha256 = "3e0b9ec325a326b0a2cef5cf164ee94a74ac372c5881ae5af634036db0441823",
    strip_prefix = "tokenizers-cpp-0.1.1",
    url = "https://github.com/mlc-ai/tokenizers-cpp/archive/refs/tags/v0.1.1.tar.gz",
)

# Android rules. Need latest rules_android_ndk to use NDK 26+.
load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")

android_ndk_repository(name = "androidndk")

# Configure Android NDK only when ANDROID_NDK_HOME is set.
# Creates current_android_ndk_env.bzl as a workaround since shell environment is available only
# through repository rule's context.
load("//:android_ndk_env.bzl", "check_android_ndk_env")

check_android_ndk_env(name = "android_ndk_env")

load("@android_ndk_env//:current_android_ndk_env.bzl", "ANDROID_NDK_HOME_IS_SET")

# Use "@android_ndk_env//:all" as a dummy toolchain target as register_toolchains() does not take
# an empty string.
register_toolchains("@androidndk//:all" if ANDROID_NDK_HOME_IS_SET else "@android_ndk_env//:all")

# VENDOR SDKS ######################################################################################

# QUALCOMM ---------------------------------------------------------------------------------------

# The actual macro call will be set during configure for now.
load("@litert//third_party/qairt:workspace.bzl", "qairt")

qairt()

# MEDIATEK ---------------------------------------------------------------------------------------

# Currently only works with local sdk
load("@litert//third_party/neuro_pilot:workspace.bzl", "neuro_pilot")

neuro_pilot()

# GOOGLE TENSOR ----------------------------------------------------------------------------------
load("@litert//third_party/google_tensor:workspace.bzl", "google_tensor")

google_tensor()

##### header-only deps for our server #########################################
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# cpp-httplib – single header :contentReference[oaicite:1]{index=1}
http_archive(
    name = "com_github_yhirose_cpp_httplib",
    urls = ["https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.tar.gz"],
    strip_prefix = "cpp-httplib-0.15.3",
    build_file_content = """
cc_library(
    name = "cpp_httplib",
    hdrs = ["httplib.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
)

# --- nlohmann/json (uses its own BUILD.bazel) ------------------------------
http_archive(
    name = "com_github_nlohmann_json",
    urls = ["https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz"],
    strip_prefix = "json-3.11.3",
)
# ---------------------------------------------------------------------------
http_archive(
    name = "com_github_jarro2783_cxxopts",
    urls = ["https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.tar.gz"],
    strip_prefix = "cxxopts-3.2.0",
    build_file_content = """
cc_library(
    name = "cxxopts",
    hdrs = glob(["include/cxxopts.hpp"]),
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
""",
)
