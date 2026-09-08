// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <acl/acl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tent/common/config.h"
#include "tent/runtime/control_plane.h"
#include "tent/transport/ascend/ascend_direct_transport.h"
#include "tent/transport/ascend/hixl_engine.h"
#include "tent/transport/ascend/resource_config.h"

namespace {

thread_local int g_device_id = 0;
thread_local aclrtContext g_context = reinterpret_cast<aclrtContext>(0x10000);
int g_device_count = 1;

constexpr uintptr_t kContextBase = 0x10000;

aclrtContext ContextForDevice(int device_id) {
    return reinterpret_cast<aclrtContext>(kContextBase +
                                          static_cast<uintptr_t>(device_id));
}

int DeviceForContext(aclrtContext context) {
    const auto value = reinterpret_cast<uintptr_t>(context);
    if (value >= kContextBase && value < kContextBase + 1024) {
        return static_cast<int>(value - kContextBase);
    }
    return -1;
}

void ResetAcl() {
    g_device_id = 0;
    g_device_count = 1;
    g_context = ContextForDevice(0);
}

}  // namespace

extern "C" {

aclError aclrtGetDeviceCount(uint32_t* count) {
    *count = static_cast<uint32_t>(g_device_count);
    return ACL_ERROR_NONE;
}

aclError aclrtGetDevice(int* deviceId) {
    *deviceId = g_device_id;
    return ACL_ERROR_NONE;
}

aclError aclrtSetDevice(int deviceId) {
    if (deviceId >= g_device_count) {
        return 1;
    }
    g_device_id = deviceId;
    g_context = ContextForDevice(deviceId);
    return ACL_ERROR_NONE;
}

aclError aclrtGetCurrentContext(aclrtContext* context) {
    *context = g_context;
    return ACL_ERROR_NONE;
}

aclError aclrtSetCurrentContext(aclrtContext context) {
    g_context = context;
    const int device_id = DeviceForContext(context);
    if (device_id >= 0) {
        g_device_id = device_id;
    }
    return ACL_ERROR_NONE;
}

aclError aclrtPointerGetAttributes(const void* ptr,
                                   aclrtPtrAttributes* attributes) {
    (void)ptr;
    attributes->location.id = 0;
    attributes->location.type = ACL_MEM_LOCATION_TYPE_HOST;
    return ACL_ERROR_NONE;
}

aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count,
                     aclrtMemcpyKind kind) {
    (void)dst;
    (void)destMax;
    (void)src;
    (void)count;
    (void)kind;
    return ACL_ERROR_NONE;
}

const char* aclGetRecentErrMsg() { return "mock acl error message"; }

aclError aclrtGetPhyDevIdByLogicDevId(int32_t logic_dev_id,
                                      int32_t* physical_dev_id) {
    *physical_dev_id = logic_dev_id;
    return ACL_ERROR_NONE;
}

}  // extern "C"

namespace mooncake {
namespace tent {
namespace {

const char* kEnvKeys[] = {
    "ASCEND_TRANSFER_TIMEOUT",
    "ASCEND_AGENT_MODE",
    "ASCEND_LOCAL_COMM_RES",
    "ASCEND_ENABLE_USE_FABRIC_MEM",
    "ASCEND_GLOBAL_RESOURCE_CONFIG",
    "ASCEND_RDMA_TC",
    "ASCEND_RDMA_SL",
    "HCCL_RDMA_TC",
    "HCCL_RDMA_SL",
    "HCCL_INTRA_ROCE_ENABLE",
    "ASCEND_BASE_PORT",
};

void ClearAscendEnv() {
    for (const char* key : kEnvKeys) {
        unsetenv(key);
    }
}

struct MockHixlState {
    std::mutex mu;
    int initialize_calls = 0;
    int connect_calls = 0;
    int disconnect_calls = 0;
    int transfer_async_calls = 0;
    int get_status_calls = 0;
    int register_calls = 0;
    int deregister_calls = 0;
    int concurrent_transfer = 0;
    int max_concurrent_transfer = 0;
    bool fail_next_transfer = false;
    int fail_register_on = -1;
    HixlXferState xfer_state = HixlXferState::Completed;
    std::map<std::string, std::string> last_options;
    std::string last_remote;
    std::vector<std::map<std::string, std::string>> all_options;
    std::atomic<uintptr_t> next_handle{0x2000};
};

class MockHixlVendor final : public HixlEngine::Vendor {
   public:
    explicit MockHixlVendor(std::shared_ptr<MockHixlState> state)
        : state_(std::move(state)) {}

    uint32_t Initialize(
        const std::string& local_name,
        const std::map<std::string, std::string>& options) override {
        (void)local_name;
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->initialize_calls++;
        state_->last_options = options;
        state_->all_options.push_back(options);
        return kHixlOk;
    }

    void Finalize() override {}

    uint32_t RegisterMem(uint64_t addr, size_t len, bool host,
                         void*& handle) override {
        (void)addr;
        (void)len;
        (void)host;
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->register_calls++;
        if (state_->fail_register_on == state_->register_calls) {
            handle = nullptr;
            return 1;
        }
        handle = reinterpret_cast<void*>(
            state_->next_handle.fetch_add(1, std::memory_order_relaxed));
        return kHixlOk;
    }

    uint32_t DeregisterMem(void* handle) override {
        (void)handle;
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->deregister_calls++;
        return kHixlOk;
    }

    uint32_t TransferAsync(const std::string& remote, bool write,
                           const std::vector<HixlOpDesc>& descs,
                           void*& req) override {
        (void)write;
        (void)descs;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            state_->transfer_async_calls++;
            state_->concurrent_transfer++;
            state_->max_concurrent_transfer = std::max(
                state_->max_concurrent_transfer, state_->concurrent_transfer);
            state_->last_remote = remote;
            if (state_->fail_next_transfer) {
                state_->fail_next_transfer = false;
                state_->concurrent_transfer--;
                req = nullptr;
                return 1;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            state_->concurrent_transfer--;
            req = reinterpret_cast<void*>(
                state_->next_handle.fetch_add(1, std::memory_order_relaxed));
        }
        return kHixlOk;
    }

    uint32_t GetTransferStatus(void* req, HixlXferState& status) override {
        (void)req;
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->get_status_calls++;
        status = state_->xfer_state;
        return kHixlOk;
    }

    uint32_t Disconnect(const std::string& remote,
                        int32_t timeout_ms) override {
        (void)remote;
        (void)timeout_ms;
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->disconnect_calls++;
        return kHixlOk;
    }

    uint32_t Connect(const std::string& remote, int32_t timeout_ms) override {
        (void)remote;
        (void)timeout_ms;
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->connect_calls++;
        return kHixlOk;
    }

   private:
    std::shared_ptr<MockHixlState> state_;
};

class AscendDirectTransportTest : public ::testing::Test {
   protected:
    void SetUp() override {
        ClearAscendEnv();
        ResetAcl();
        state_ = std::make_shared<MockHixlState>();
        HixlEngine::SetVendorFactory([state = state_]() {
            return std::make_unique<MockHixlVendor>(state);
        });
        metadata_ = std::make_shared<ControlService>("p2p", "", nullptr);
        transport_ = std::make_unique<AscendDirectTransport>();
    }

    void TearDown() override {
        if (batch_ != nullptr) {
            (void)transport_->freeSubBatch(batch_);
        }
        if (transport_) {
            (void)transport_->uninstall();
        }
        HixlEngine::SetVendorFactory(nullptr);
        ClearAscendEnv();
    }

    Status Install(std::shared_ptr<Config> conf = nullptr) {
        name_ = "127.0.0.1:23456";
        return transport_->install(name_, metadata_, nullptr, conf);
    }

    // Fabric mem keeps the HIXL TransferAsync path; same-engine copies
    // otherwise use aclrtMemcpy because AutoConnect rejects self.
    Status InstallHixlPath(std::shared_ptr<Config> conf = nullptr) {
        if (!conf) {
            conf = std::make_shared<Config>();
        }
        conf->set("transports/ascend_direct/fabric_mem", true);
        return Install(conf);
    }

    Request MakeWrite(void* src, uint64_t dest, size_t len) {
        Request request{};
        request.opcode = Request::WRITE;
        request.source = src;
        request.target_id = LOCAL_SEGMENT_ID;
        request.target_offset = dest;
        request.length = len;
        return request;
    }

    std::shared_ptr<MockHixlState> state_;
    std::shared_ptr<ControlService> metadata_;
    std::unique_ptr<AscendDirectTransport> transport_;
    Transport::SubBatchRef batch_{nullptr};
    std::string name_;
};

TEST_F(AscendDirectTransportTest, InitializeIsAutoConnectOnly) {
    ASSERT_TRUE(Install().ok());
    const auto& opts = AscendDirectTransportTestPeer::initOptions(*transport_);
    EXPECT_EQ(opts.at("AutoConnect"), "1");
    EXPECT_EQ(opts.at("LocalCommRes"), kDefaultLocalCommRes);
    EXPECT_EQ(opts.count("BufferPool"), 0u);
    EXPECT_EQ(opts.count("ShortConnection"), 0u);
    EXPECT_EQ(state_->connect_calls, 0);
    EXPECT_FALSE(transport_->supportNotification());
}

TEST_F(AscendDirectTransportTest, LocalCommResEnvOverride) {
    setenv("ASCEND_LOCAL_COMM_RES", R"({"version":"9.9"})", 1);
    ASSERT_TRUE(Install().ok());
    const auto& opts = AscendDirectTransportTestPeer::initOptions(*transport_);
    EXPECT_EQ(opts.at("LocalCommRes"), R"({"version":"9.9"})");
}

TEST_F(AscendDirectTransportTest, FabricOptionInjectedFromConfig) {
    auto conf = std::make_shared<Config>();
    conf->set("transports/ascend_direct/fabric_mem", true);
    ASSERT_TRUE(Install(conf).ok());
    const auto& opts = AscendDirectTransportTestPeer::initOptions(*transport_);
    EXPECT_EQ(opts.at("EnableUseFabricMem"), "1");
    EXPECT_TRUE(
        AscendDirectTransportTestPeer::options(*transport_).use_fabric_mem);
}

TEST_F(AscendDirectTransportTest, FabricEnvDoesNotLeakWithoutStoreTeInit) {
    setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "1", 1);
    auto options = LoadAscendDirectOptions(nullptr);
    EXPECT_FALSE(options.use_fabric_mem);
    auto conf = std::make_shared<Config>();
    conf->set("transports/ascend_direct/store_te_init", true);
    options = LoadAscendDirectOptions(conf);
    EXPECT_TRUE(options.use_fabric_mem);
}

TEST_F(AscendDirectTransportTest, TimeoutMsFromEnv) {
    setenv("ASCEND_TRANSFER_TIMEOUT", "2500", 1);
    auto options = LoadAscendDirectOptions(nullptr);
    EXPECT_EQ(options.transfer_timeout_ms, 2500);
}

TEST_F(AscendDirectTransportTest, BuildHixlInitOptionsNeverSetsBufferPool) {
    AscendDirectOptions options;
    options.use_fabric_mem = true;
    options.rdma_tc = "192";
    auto init = BuildHixlInitOptions(options);
    EXPECT_EQ(init.at("AutoConnect"), "1");
    EXPECT_EQ(init.at("EnableUseFabricMem"), "1");
    EXPECT_EQ(init.at("RdmaTrafficClass"), "192");
    EXPECT_EQ(init.count("BufferPool"), 0u);
    EXPECT_EQ(init.count("Connect"), 0u);
}

TEST_F(AscendDirectTransportTest, InitializeInjectsPerEngineListenPort) {
    setenv("ASCEND_GLOBAL_RESOURCE_CONFIG",
           R"({"comm_resource_config":{"protocol_desc":"roce:device"}})", 1);
    ASSERT_TRUE(Install().ok());
    ASSERT_EQ(state_->last_options.count("GlobalResourceConfig"), 1u);
    const auto& cfg = state_->last_options.at("GlobalResourceConfig");
    EXPECT_NE(cfg.find("listen_port"), std::string::npos);
    EXPECT_NE(cfg.find("roce:device"), std::string::npos);
}

TEST_F(AscendDirectTransportTest, SubmitRollbackLeavesBatchUnchanged) {
    ASSERT_TRUE(Install().ok());
    ASSERT_TRUE(transport_->allocateSubBatch(batch_, 8).ok());
    char src[64]{};
    std::vector<Request> requests = {MakeWrite(src, 0x1000, 64),
                                     MakeWrite(src, 0x2000, 64)};
    requests[1].target_id = 99;
    auto status = transport_->submitTransferTasks(batch_, requests);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(batch_->size(), 0u);
    EXPECT_EQ(state_->transfer_async_calls, 0);
}

TEST_F(AscendDirectTransportTest, LocalCopySameEngineSkipsTransferAsync) {
    ASSERT_TRUE(Install().ok());
    ASSERT_TRUE(transport_->allocateSubBatch(batch_, 4).ok());
    char src[32]{};
    ASSERT_TRUE(
        transport_->submitTransferTasks(batch_, {MakeWrite(src, 0x1000, 32)})
            .ok());
    TransferStatus status{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_, 0, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.transferred_bytes, 32u);
    EXPECT_EQ(state_->transfer_async_calls, 0);
    EXPECT_EQ(state_->connect_calls, 0);
    EXPECT_EQ(state_->disconnect_calls, 0);
    EXPECT_EQ(AscendDirectTransportTestPeer::inflight(*transport_), 0);
}

TEST_F(AscendDirectTransportTest, TransferAsyncSuccessAndNoConnect) {
    ASSERT_TRUE(InstallHixlPath().ok());
    ASSERT_TRUE(transport_->allocateSubBatch(batch_, 4).ok());
    char src[32]{};
    ASSERT_TRUE(
        transport_->submitTransferTasks(batch_, {MakeWrite(src, 0x1000, 32)})
            .ok());
    TransferStatus status{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_, 0, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.transferred_bytes, 32u);
    EXPECT_EQ(state_->transfer_async_calls, 1);
    EXPECT_EQ(state_->connect_calls, 0);
    EXPECT_EQ(state_->disconnect_calls, 0);
    EXPECT_EQ(AscendDirectTransportTestPeer::inflight(*transport_), 0);
}

TEST_F(AscendDirectTransportTest, FailEntireRouteMarksSiblingTasks) {
    state_->xfer_state = HixlXferState::Failed;
    ASSERT_TRUE(InstallHixlPath().ok());
    ASSERT_TRUE(transport_->allocateSubBatch(batch_, 4).ok());
    char src[64]{};
    ASSERT_TRUE(transport_
                    ->submitTransferTasks(batch_, {MakeWrite(src, 0x1000, 32),
                                                   MakeWrite(src, 0x2000, 32)})
                    .ok());
    TransferStatus first{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_, 0, first).ok());
    EXPECT_EQ(first.s, TransferStatusEnum::FAILED);
    const int polls_after_first = state_->get_status_calls;
    TransferStatus second{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_, 1, second).ok());
    EXPECT_EQ(second.s, TransferStatusEnum::FAILED);
    EXPECT_EQ(state_->get_status_calls, polls_after_first);
    EXPECT_EQ(state_->disconnect_calls, 0);
}

TEST_F(AscendDirectTransportTest, TimeoutDisconnectsHungRoute) {
    state_->xfer_state = HixlXferState::Waiting;
    auto conf = std::make_shared<Config>();
    conf->set("transports/ascend_direct/transfer_timeout_ms", 1);
    ASSERT_TRUE(InstallHixlPath(conf).ok());
    ASSERT_TRUE(transport_->allocateSubBatch(batch_, 2).ok());
    char src[16]{};
    ASSERT_TRUE(
        transport_->submitTransferTasks(batch_, {MakeWrite(src, 0x1000, 16)})
            .ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    TransferStatus status{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_, 0, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::TIMEOUT);
    EXPECT_GE(state_->disconnect_calls, 1);
    EXPECT_EQ(state_->connect_calls, 0);
}

TEST_F(AscendDirectTransportTest, MultiEnginePublishesNamesAndRoutesDest) {
    g_device_count = 2;
    auto conf = std::make_shared<Config>();
    conf->set("transports/ascend_direct/agent_mode", true);
    ASSERT_TRUE(Install(conf).ok());
    ASSERT_EQ(AscendDirectTransportTestPeer::engineCount(*transport_), 2u);
    const auto name0 =
        AscendDirectTransportTestPeer::engineName(*transport_, 0);
    const auto name1 =
        AscendDirectTransportTestPeer::engineName(*transport_, 1);
    EXPECT_NE(name0, name1);

    auto local = metadata_->segmentManager().getLocal();
    const auto& detail = std::get<MemorySegmentDesc>(local->detail);
    EXPECT_EQ(detail.device_attrs.at(kHixlNameAttr), name0);
    auto names = ParseHixlNames(detail);
    ASSERT_EQ(names.size(), 2u);

    BufferDesc buf0{};
    buf0.addr = 0x10000;
    buf0.length = 4096;
    buf0.transport_attrs[AscendDirect] = "0";
    BufferDesc buf1{};
    buf1.addr = 0x20000;
    buf1.length = 4096;
    buf1.transport_attrs[AscendDirect] = "1";
    ASSERT_TRUE(metadata_->segmentManager()
                    .updateLocal([&](SegmentDesc& segment) {
                        auto& mem = std::get<MemorySegmentDesc>(segment.detail);
                        mem.buffers.push_back(buf0);
                        mem.buffers.push_back(buf1);
                        return Status::OK();
                    })
                    .ok());

    ASSERT_TRUE(transport_->allocateSubBatch(batch_, 2).ok());
    char src[32]{};
    ASSERT_TRUE(
        transport_->submitTransferTasks(batch_, {MakeWrite(src, 0x20010, 16)})
            .ok());
    EXPECT_EQ(state_->last_remote, name1);
    TransferStatus status{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_, 0, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
}

TEST_F(AscendDirectTransportTest, RegisterRollbackUnregistersPrefix) {
    ASSERT_TRUE(Install().ok());
    state_->fail_register_on = 2;
    std::vector<BufferDesc> desc_list(2);
    desc_list[0].addr = 0x1000;
    desc_list[0].length = 64;
    desc_list[0].location = "cpu:0";
    desc_list[1].addr = 0x2000;
    desc_list[1].length = 64;
    desc_list[1].location = "cpu:0";
    MemoryOptions options;
    auto status = transport_->addMemoryBuffer(desc_list, options);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(state_->register_calls, 2);
    EXPECT_EQ(state_->deregister_calls, 1);
}

TEST_F(AscendDirectTransportTest, QuiesceWaitsForInflight) {
    state_->xfer_state = HixlXferState::Waiting;
    auto conf = std::make_shared<Config>();
    conf->set("transports/ascend_direct/transfer_timeout_ms", 50);
    ASSERT_TRUE(InstallHixlPath(conf).ok());
    ASSERT_TRUE(transport_->allocateSubBatch(batch_, 2).ok());
    char src[16]{};
    ASSERT_TRUE(
        transport_->submitTransferTasks(batch_, {MakeWrite(src, 0x1000, 16)})
            .ok());
    EXPECT_EQ(AscendDirectTransportTestPeer::inflight(*transport_), 1);
    state_->xfer_state = HixlXferState::Completed;
    TransferStatus status{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_, 0, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    EXPECT_TRUE(transport_->quiesce().ok());
    EXPECT_EQ(AscendDirectTransportTestPeer::inflight(*transport_), 0);
}

TEST_F(AscendDirectTransportTest, PerEngineMutexSerializesSubmit) {
    ASSERT_TRUE(InstallHixlPath().ok());
    char src_a[16]{};
    char src_b[16]{};
    Transport::SubBatchRef batch_a = nullptr;
    Transport::SubBatchRef batch_b = nullptr;
    ASSERT_TRUE(transport_->allocateSubBatch(batch_a, 2).ok());
    ASSERT_TRUE(transport_->allocateSubBatch(batch_b, 2).ok());
    std::thread t1([&] {
        (void)transport_->submitTransferTasks(batch_a,
                                              {MakeWrite(src_a, 0x1000, 16)});
    });
    std::thread t2([&] {
        (void)transport_->submitTransferTasks(batch_b,
                                              {MakeWrite(src_b, 0x2000, 16)});
    });
    t1.join();
    t2.join();
    EXPECT_EQ(state_->max_concurrent_transfer, 1);
    EXPECT_EQ(state_->connect_calls, 0);
    TransferStatus status_a{};
    TransferStatus status_b{};
    ASSERT_TRUE(transport_->getTransferStatus(batch_a, 0, status_a).ok());
    ASSERT_TRUE(transport_->getTransferStatus(batch_b, 0, status_b).ok());
    EXPECT_EQ(status_a.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status_b.s, TransferStatusEnum::COMPLETED);
    (void)transport_->freeSubBatch(batch_a);
    (void)transport_->freeSubBatch(batch_b);
}

TEST(AscendDirectResourceConfigTest, StoreJsonIsolation) {
    ClearAscendEnv();
    auto conf = std::make_shared<Config>();
    conf->set("transports/ascend_direct/global_resource_config",
              R"({"store":{"fabric_memory":{"a":1}},"other":1})");
    auto p2p = LoadAscendDirectOptions(conf);
    EXPECT_FALSE(p2p.use_fabric_mem);
    conf->set("transports/ascend_direct/store_te_init", true);
    auto store = LoadAscendDirectOptions(conf);
    EXPECT_TRUE(store.use_fabric_mem);
}

TEST(AscendDirectResourceConfigTest, DestRoutingUsesBufferEngineIndex) {
    MemorySegmentDesc detail;
    detail.device_attrs[kHixlNamesAttr] = R"(["a:1","b:2"])";
    BufferDesc buf{};
    buf.addr = 100;
    buf.length = 50;
    buf.transport_attrs[AscendDirect] = "1";
    detail.buffers.push_back(buf);
    EXPECT_EQ(ResolveRemoteHixlName(detail, 120), "b:2");
}

TEST(AscendDirectResourceConfigTest, RoceListenPortAvoidsHixlControlPort) {
    EXPECT_EQ(HixlRoceListenPort(20000), 30000);
    EXPECT_EQ(HixlRoceListenPort(42067), 52067);
    EXPECT_NE(HixlRoceListenPort(20000), 20000u);
    EXPECT_EQ(HixlRoceListenPort(60000), 50000);
}

TEST(AscendDirectResourceConfigTest, WithHixlListenPortKeepsProtocolDesc) {
    AscendDirectOptions options;
    options.global_resource_config =
        R"({"comm_resource_config":{"protocol_desc":"roce:device"}})";
    auto init = WithHixlListenPort(BuildHixlInitOptions(options), 52067);
    const auto& cfg = init.at("GlobalResourceConfig");
    EXPECT_NE(cfg.find("\"listen_port\":52067"), std::string::npos);
    EXPECT_NE(cfg.find("roce:device"), std::string::npos);
}

}  // namespace
}  // namespace tent
}  // namespace mooncake
