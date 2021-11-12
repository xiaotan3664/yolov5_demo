//
// Created by yuan on 1/22/21.
//

#ifndef YOLOV5_DEMO_BMNN_UTILS_H
#define YOLOV5_DEMO_BMNN_UTILS_H

#include <iostream>
#include <string>
#include <memory>

#include "bmruntime_interface.h"
#include "bm_wrapper.hpp"

class NoCopyable {
protected:
    NoCopyable() =default;
    ~NoCopyable() = default;
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable& rhs)= delete;
};

class BMNNTensor{
    /**
     *  members from bm_tensor {
     *  bm_data_type_t dtype;
        bm_shape_t shape;
        bm_device_mem_t device_mem;
        bm_store_mode_t st_mode;
        }
     */
    bm_handle_t  m_handle;

    std::string m_name;
    float *m_cpu_data;
    float m_scale;
    bm_tensor_t *m_tensor;

public:
    BMNNTensor(bm_handle_t handle, const char *name, float scale,
            bm_tensor_t* tensor):m_handle(handle), m_name(name),
            m_cpu_data(nullptr),m_scale(scale), m_tensor(tensor) {
    }

    virtual ~BMNNTensor() {
        if (m_cpu_data != NULL) {
            delete [] m_cpu_data;
            m_cpu_data = NULL;
        }
    }

    int set_device_mem(bm_device_mem_t *mem){
        this->m_tensor->device_mem = *mem;
        return 0;
    }

    const bm_device_mem_t* get_device_mem() {
        return &this->m_tensor->device_mem;
    }

    float *get_cpu_data() {
        if (m_cpu_data == NULL) {
            bm_status_t ret;
            float *pFP32 = nullptr;
            int count = bmrt_shape_count(&m_tensor->shape);
            if (m_tensor->dtype == BM_FLOAT32) {
                pFP32 = new float[count];
                assert(pFP32 != nullptr);
                ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem, count * sizeof(float));
                assert(BM_SUCCESS ==ret);
            }else if (BM_INT8 == m_tensor->dtype) {
                int tensor_size = bmrt_tensor_bytesize(m_tensor);
                int8_t *pU8 = new int8_t[tensor_size];
                assert(pU8 != nullptr);
                pFP32 = new float[count];
                assert(pFP32 != nullptr);
                ret = bm_memcpy_d2s_partial(m_handle, pU8, m_tensor->device_mem, tensor_size);
                assert(BM_SUCCESS ==ret);
                for(int i = 0;i < count; ++ i) {
                    pFP32[i] = pU8[i] * m_scale;
                }
                delete [] pU8;
            }else{
                std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
            }

            m_cpu_data = pFP32;
        }

        return m_cpu_data;
    }

    const bm_shape_t* get_shape() {
        return &m_tensor->shape;
    }

    bm_data_type_t get_dtype() {
        return m_tensor->dtype;
    }

    float get_scale() {
        return m_scale;
    }

};

class BMNNNetwork : public NoCopyable {
    const bm_net_info_t *m_netinfo;
    bm_tensor_t* m_inputTensors;
    bm_tensor_t* m_outputTensors;
    bm_handle_t  m_handle;
    void *m_bmrt;

    std::unordered_map<std::string, bm_tensor_t*> m_mapInputs;
    std::unordered_map<std::string, bm_tensor_t*> m_mapOutputs;

public:
    BMNNNetwork(void *bmrt, const std::string& name):m_bmrt(bmrt) {
        m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
        m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
        m_inputTensors = new bm_tensor_t[m_netinfo->input_num];
        m_outputTensors = new bm_tensor_t[m_netinfo->output_num];
        for(int i = 0; i < m_netinfo->input_num; ++i) {
            m_inputTensors[i].dtype = m_netinfo->input_dtypes[i];
            m_inputTensors[i].shape = m_netinfo->stages[0].input_shapes[i];
            m_inputTensors[i].st_mode = BM_STORE_1N;
            m_inputTensors[i].device_mem = bm_mem_null();
        }

        for(int i = 0; i < m_netinfo->output_num; ++i) {
            m_outputTensors[i].dtype = m_netinfo->output_dtypes[i];
            m_outputTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
            m_outputTensors[i].st_mode = BM_STORE_1N;
            m_outputTensors[i].device_mem = bm_mem_null();
        }

        assert(m_netinfo->stage_num == 1);
    }

    ~BMNNNetwork() {
        //Free input tensors
        delete [] m_inputTensors;
        //Free output tensors
        for(int i = 0; i < m_netinfo->output_num; ++i) {
            if (m_outputTensors[i].device_mem.size != 0) {
                bm_free_device(m_handle, m_outputTensors[i].device_mem);
            }
        }
        delete []m_outputTensors;
    }

    int inputTensorNum() {
        return m_netinfo->input_num;
    }

    std::shared_ptr<BMNNTensor> inputTensor(int index){
        assert(index < m_netinfo->input_num);
        return std::make_shared<BMNNTensor>(m_handle, m_netinfo->input_names[index],
                m_netinfo->input_scales[index], &m_inputTensors[index]);
    }

    int outputTensorNum() {
        return m_netinfo->output_num;
    }

    std::shared_ptr<BMNNTensor> outputTensor(int index){
        assert(index < m_netinfo->output_num);
        return std::make_shared<BMNNTensor>(m_handle, m_netinfo->output_names[index],
                m_netinfo->output_scales[index], &m_outputTensors[index]);
    }

    int forward() {

        bool user_mem = false; // if false, bmrt will alloc mem every time.
        if (m_outputTensors->device_mem.size != 0) {
            // if true, bmrt don't alloc mem again.
            user_mem = true;
        }

        bool ok=bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,
                m_outputTensors, m_netinfo->output_num, user_mem, false);
        if (!ok) {
            std::cout << "bm_launch_tensor() failed=" << std::endl;
            return -1;
        }

#if 0
        for(int i = 0;i < m_netinfo->output_num; ++i) {
            auto tensor = m_outputTensors[i];
            // dump
            std::cout << "output_tensor [" << i << "] size=" << bmrt_tensor_device_size(&tensor) << std::endl;
        }
#endif

        return 0;
    }



};

class BMNNHandle: public NoCopyable {
    bm_handle_t m_handle;
    int m_dev_id;
public:
    BMNNHandle(int dev_id=0):m_dev_id(dev_id) {
        int ret = bm_dev_request(&m_handle, dev_id);
        assert(BM_SUCCESS == ret);
    }

    ~BMNNHandle(){
        bm_dev_free(m_handle);
    }

    bm_handle_t handle() {
        return m_handle;
    }

    int dev_id() {
        return m_dev_id;
    }
};

using BMNNHandlePtr = std::shared_ptr<BMNNHandle>;

class BMNNContext : public NoCopyable {
    BMNNHandlePtr m_handlePtr;
    void *m_bmrt;
    std::vector<std::string> m_network_names;

public:
    BMNNContext(BMNNHandlePtr handle, const char* bmodel_file):m_handlePtr(handle){
         bm_handle_t hdev = m_handlePtr->handle();
         m_bmrt = bmrt_create(hdev);
         if (NULL == m_bmrt) {
             std::cout << "bmrt_create() failed!" << std::endl;
             exit(-1);
         }

         if (!bmrt_load_bmodel(m_bmrt, bmodel_file)) {
             std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
         }

         load_network_names();


    }

    ~BMNNContext() {
        if (m_bmrt!=NULL) {
            bmrt_destroy(m_bmrt);
            m_bmrt = NULL;
        }
    }

    bm_handle_t handle() {
        return m_handlePtr->handle();
    }

    void* bmrt() {
        return m_bmrt;
    }

    void load_network_names() {
        const char **names;
        int num;
        num = bmrt_get_network_number(m_bmrt);
        bmrt_get_network_names(m_bmrt, &names);
        for(int i=0;i < num; ++i) {
            m_network_names.push_back(names[i]);
        }

        free(names);
    }

    std::string network_name(int index){
        if (index >= (int)m_network_names.size()) {
           return "Invalid index";
        }

        return m_network_names[index];
    }

    std::shared_ptr<BMNNNetwork> network(const std::string& net_name)
    {
        return std::make_shared<BMNNNetwork>(m_bmrt, net_name);
    }

    std::shared_ptr<BMNNNetwork> network(int net_index) {
        assert(net_index < (int)m_network_names.size());
        return std::make_shared<BMNNNetwork>(m_bmrt, m_network_names[net_index]);
    }



};







#endif //YOLOV5_DEMO_BMNN_UTILS_H
