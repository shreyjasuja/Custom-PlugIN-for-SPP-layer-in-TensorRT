#ifndef TRT_SPP_POOL_PLUGIN_H
#define TRT_SPP_POOL_PLUGIN_H
#include "plugin.h"
#include <string>
#include <vector>

// #include "NvInferPluginUtils.h"

typedef struct
{
    std::vector<int> pool_List 
}SPP_PoolingParams;

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{

class SPP_Pool : public IPluginV2Ext
{
public:
    SPP_Pool(SPP_PoolingParams params, int N, int C, int H, int W);  // int nBegPad, int nEndPad,

    SPP_Pool(const void* buffer, size_t length);

    ~SPP_Pool() override = default;

    int getNbOutputs() const override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(nvinfer1::DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
        const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;


private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        std::memcpy(buffer, &val, sizeof(T));
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val;
        std::memcpy(&val, buffer, sizeof(T));
        buffer += sizeof(T);
        return val;
    }

    SPP_PoolingParams mParams;
    int mN, mC, mH, mW;

    const char* mPluginNamespace = "";
};

class SPP_PoolPluginCreator : public BaseCreator
{
public:
    SPP_PoolPluginCreator();

    ~SPP_PoolPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    SPP_PoolingParams params;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace = "";

};

} // namespace plugin
} // namespace nvinfer1

PluginFieldCollection SPP_PoolPluginCreator::mFC{};
std::vector<PluginField> SPP_PoolPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(SPP_PoolPluginCreator);




#endif // TRT_SPP_Pool_PLUGIN_H