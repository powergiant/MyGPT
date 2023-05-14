# Distributed data parallel

## references

1. torch.nn.parallel.DistributedDataParallel(比较浅没有贴完整训练code): 快速上手: https://zhuanlan.zhihu.com/p/467103734
2. PyTorch 数据并行处理(比较浅没有贴完整训练code): https://pytorch.panchuang.net/SecondSection/optional_data_parallelism/
3. pytorch分布式数据并行DistributedDataParallel（DDP）(比较浅没有贴完整训练code): https://zhuanlan.zhihu.com/p/107139605
4. A Comprehensive Tutorial to Pytorch DistributedDataParallel(比较浅没有贴完整训练code, 但讲的还算仔细): https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
5. Distributed communication package - torch.distributed: https://pytorch.org/docs/stable/distributed.html
6. nanoGPT: https://github.com/karpathy/nanoGPT
7. YaLM: https://github.com/yandex/YaLM-100B
8. Deepspeed-Megatron: https://github.com/microsoft/Megatron-DeepSpeed
9. GPT-neox: https://github.com/EleutherAI/gpt-neox
10. 云平台: https://vast.ai https://www.coreweave.com/ https://www.fluidstack.io https://cloud.tencent.com/product/gpu https://www.autodl.com



## run

```
if __name__ == '__main__':
    torch.multiprocessing.spawn(
                trainer_ddp.train,
                args = (model, train_config, config.model_config, config.ddp_config, dataset),
                nprocs = config.ddp_config.world_size
            )
```

Must inside "if __name__ == '__main__':" or you get the error 
```
RuntimeError: 
            Attempt to start a new process before the current process
            has finished its bootstrapping phase.
```



# deepspeed