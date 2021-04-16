#include "inference.h"
#include "cnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdlib.h"
#include <sys/time.h>
#include <time.h>

namespace StyleTransfer
{

    Inference ::Inference(std::string offline_model)
    {
        offline_model_ = offline_model;
    }

    void Inference ::run(DataTransfer *DataT)
    {
        printf("input\tdata[0]\t%f\n", DataT->input_data[1 * 256 * 256 * 3 - 3]);
        printf("input\tdata[0]\t%f\n", DataT->input_data[1 * 256 * 256 * 3]);
        printf("input\tnum\t%d\n", DataT->input_num);
        printf("model_name %s\n", DataT->model_name.c_str());

        // YKH:load model
        cnrtModel_t model;
        char fname[100] = "/home/AICSE-demo-student/demo/style_transfer_bcl/models/offline_models/";
        strcat(fname, DataT->model_name.c_str());
        strcat(fname, ".cambricon");
        printf("file_path\t%s\n", fname);
        cnrtLoadModel(&model, fname);

        // YKH:set current device
        cnrtInit(0);
        cnrtDev_t dev;
        cnrtGetDeviceHandle(&dev, 0);
        cnrtSetCurrentDevice(dev);

        // YKH:load extract function
        cnrtFunction_t function;
        cnrtCreateFunction(&function);
        cnrtExtractFunction(&function, model, "subnet0");

        int inputNum, outputNum;
        int64_t *inputSizeS, *outputSizeS;
        cnrtGetInputDataSize(&inputSizeS, &inputNum, function);    // Size 每个输出的长度
        cnrtGetOutputDataSize(&outputSizeS, &outputNum, function); // Num 有几个输出
        // YKH:prepare data on cpu
        void **inputCpuPtrS = (void **)malloc(inputNum * sizeof(void *));
        void **outputCpuPtrS = (void **)malloc(outputNum * sizeof(void *));

        // YKH:allocate I/O data memory on MLU
        void **inputMluPtrS = (void **)malloc(inputNum * sizeof(void *));
        void **outputMluPtrS = (void **)malloc(outputNum * sizeof(void *));

        // YKH:prepare input buffer
        uint16_t *input_half; // NCWH格式的中间变量
        for (int i = 0; i < inputNum; i++)
        {
            // converts data format when using new interface model
            inputCpuPtrS[i] = (uint16_t *)malloc(inputSizeS[i] * 1);
            // malloc cpu memory
            input_half = (uint16_t *)malloc(inputSizeS[i] * 1);
            int length = inputSizeS[i] / 2;
            for (int j = 0; j < length; j++)
            {
                cnrtConvertFloatToHalf(input_half + j, DataT->input_data[j]);
            }
            cnrtReshapeNCHWToNHWC(inputCpuPtrS[i], input_half, 1, 256, 256, 3, cnrtDataType_t(0x12 /*FLOAT16*/));
            // malloc mlu memory
            cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]);
            cnrtMemcpy(inputMluPtrS[i], inputCpuPtrS[i], inputSizeS[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
        }

        // YKH:prepare output buffer
        float *output_temp;
        for (int i = 0; i < outputNum; i++)
        {
            outputCpuPtrS[i] = (uint16_t *)malloc(outputSizeS[i] * 1);
            // YKH:malloc cpu memory
            output_temp = new float[256 * 256 * 3];
            // YKH:malloc mlu memory
            cnrtMalloc(&(outputMluPtrS[i]), outputSizeS[i]);
        }

        // FROM DOCUMENT: prepare parameters for cnrtInvokeRuntimeContext_V2
        void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
        for (int i = 0; i < inputNum; ++i)
        {
            param[i] = inputMluPtrS[i];
        }
        for (int i = 0; i < outputNum; ++i)
        {
            param[inputNum + i] = outputMluPtrS[i];
        }

        // YKH:setup runtime ctx
        cnrtRuntimeContext_t ctx;
        cnrtCreateRuntimeContext(&ctx, function, NULL);

        // YKH:bind device
        cnrtSetRuntimeContextDeviceId(ctx, 0);
        cnrtInitRuntimeContext(ctx, NULL);

        // YKH:compute offline
        cnrtQueue_t queue;
        cnrtRuntimeContextCreateQueue(ctx, &queue);
        cnrtInvokeRuntimeContext_V2(ctx, NULL, param, queue, NULL);
        cnrtSyncQueue(queue);


        for (int i = 0; i < outputNum; i++)
        {
            // copy to cpu
            cnrtMemcpy(outputCpuPtrS[i], outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST);
            // convert to float
            int length = outputSizeS[i] / 2;
            uint16_t *outputCpu = ((uint16_t **)outputCpuPtrS)[0];
            DataT->output_data = new float[256 * 256 * 3];
            for (int j = 0; j < length; j++)
            {
                cnrtConvertHalfToFloat(output_temp + j, outputCpu[j]);
            }
            cnrtReshapeNHWCToNCHW(DataT->output_data, output_temp, 1, 256, 256, 3, cnrtDataType_t(0x13 /*FLOAT32*/));
        }

        // YKH:free memory spac
        for (int i = 0; i < inputNum; i++)
        {
            free(inputCpuPtrS[i]);
            cnrtFree(inputMluPtrS[i]);
        }
        for (int i = 0; i < outputNum; i++)
        {
            free(outputCpuPtrS[i]);
            cnrtFree(outputMluPtrS[i]);
        }
        free(inputCpuPtrS);
        free(outputCpuPtrS);
        free(param);
        free(input_half);
        delete output_temp;

        cnrtDestroyQueue(queue);
        cnrtDestroyRuntimeContext(ctx);
        cnrtDestroyFunction(function);
        cnrtUnloadModel(model);
        cnrtDestroy();
        std::cout << "The run time is: " <<(double)clock() / CLOCKS_PER_SEC << "s" << std::endl;
    }

} // namespace StyleTransfer
