#include "SPP_Pool.h"
#include "NvInferPlugin.h"
#include "averagePool.h"
#include <cassert>
#include <string.h>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

#define ASSERT assert

namespace 
{
    constexpr const char* SPP_POOL_PLUGIN_NAME{"SPP_Pool_TRT"};
    constexpr const char* SPP_POOL_PLUGIN_VERSION{"001"};
}


SPP_Pool::SPP_Pool(SPP_PoolingParams params, int N,int C,int H,int W )
	,mParams(params)
    , mN(N)
    , mC(C)
    , mH(H)
    , mW(W)
{

}

SPP_Pool::SPP_Pool(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;

    mN = read<int>(d);
    mC = read<int>(d);
    mH = read<int>(d);
    mW = read<int>(d);
}

int SPP_Pool::getNbOutputs() const
{
    // We always return one output
    return 1;
}

nvinfer1::Dims SPP_Pool::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    return *inputs;
}

int SPP_Pool::initialize()
{
    return 0;
}

void SPP_Pool::terminate()
{
}

size_t SPP_Pool::getWorkspaceSize(int maxBatchSize) const
{
    // The operation is done in place, it doesn't use GPU memory
    return 0;
}

int SPP_Pool::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    return avgPool(batchSize,
                   static_cast<const float *>(inputs[0]),
                   static_cast<float *>(outputs[0]),
                   mC,
                   mH,
                   mW,
                   stream);
}

void SPP_Pool::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mN);
    write(d, mC);
    write(d, mH);
    write(d, mW);
    ASSERT(d == a + getSerializationSize());
}
size_t S3Pool::getSerializationSize() const
{
    // N, C, H, W => 4
    return  (4 + pool_list.size()) * sizeof(int);                    
}
// Set plugin namespace
void SPP_Pool::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* SPP_Pool::getPluginNamespace() const
{
    return mPluginNamespace;
}

void SPP_Pool::configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
    const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{

    ASSERT(inputDims->nbDims == 3);
    // Configured with batch size = 1
    mN = 1;
    mC = inputDims->d[0];
    mH = inputDims->d[1];
    mW = inputDims->d[2];
}


// Return the nvinfer1::DataType of the plugin output at the requested index
DataType SPP_Pool::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index == 0);
    return nvinfer1::DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool SPP_Pool::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool SPP_Pool::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

bool S3Pool::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == nvinfer1::DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* SPP_Pool::getPluginType() const
{
    return SPP_POOL_PLUGIN_NAME;
}

void SPP_Pool::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

void SPP_Pool::detachFromContext()
{
}

void SPP_Pool::destroy()
{
    delete this;
}

const char* SPP_Pool::getPluginVersion() const
{
    return SPP_POOL_PLUGIN_VERSION;
}

IPluginV2Ext* SPP_Pool::clone() const
{
    return new SPP_Pool(mN, mC, mH, mW);
}

SPP_PoolPluginCreator::SPP_PoolPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();

}

const char* SPP_PoolPluginCreator::getPluginName() const
{
    return SPP_POOL_PLUGIN_NAME;
}

const char* SPP_PoolPluginCreator::getPluginVersion() const
{
    return SPP_POOL_PLUGIN_VERSION;
}

const PluginFieldCollection* SPP_PoolPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* SPP_PoolPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for(int i=0;i<pool_list.size();i++)
    	{ 
    		row_length=mH/pool_list[i];
    		col_length=mW/pool_list[i];

    	for(int pool_num=0; pool_num<pool_list.size();pool_num++)
     		{
     			int num_pool_region=pool_list[pool_num]

                	for(int jy=0; jy<num_pool_regions;jy++)
                    	{
                    		for(int ix=0;ix<num_pool_regions;ix++)
                    		{	
                        		x1 = ix * row_length[pool_num]
                        		x2 = ix * row_length[pool_num] + row_length[pool_num]
                        		y1 = jy * col_length[pool_num]
                        		y2 = jy * col_length[pool_num] + col_length[pool_num]

                        		x1 =int(round(x1))
                        		x2 =int(round(x2))
                        		y1 =int(round(y1))
                        		y2 =int(round(y2))
                        		new_shape = [mN,mC, y2 - y1, x2 - x1]
                        		x_crop = x[:, :, y1:y2, x1:x2]
                        													//xm = K.reshape(x_crop, new_shape)
                        													//pooled_val = K.max(xm, axis=(2, 3))
                        													//outputs.append(pooled_val)
                        	}
                    	}
        	}
        }

    SPP_Pool *obj = new SPP_Pool(/*params, params.kernel_shape[0],params.strides[0]*/);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* SPP_PoolPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    SPP_Pool *obj = new SPP_Pool(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}