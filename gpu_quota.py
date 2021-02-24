import pynvml
import os, time
pynvml.nvmlInit()
# GPU id = 0
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
ratio = 1024**2
while 1:
    # Sleep for 60s
    time.sleep(60)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    #MB: devided by 1024**2
    total = meminfo.total/ratio 
    used = meminfo.used/ratio
    free = meminfo.free/ratio
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("{}, total: {}MB, used: {}MB, free: {}MB".format(t, int(total), int(used), int(free)))
    
    if used < total/8:
        print("start")
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py -num_workers 8 -test_with fake-real')
        print("finish!")
        break