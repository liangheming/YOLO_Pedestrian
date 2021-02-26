from solver.ddp_mix_solver import DDPMixSolver

# from processors.ddp_mix_processorv4 import COCODDPMixProcessor
# python -m torch.distributed.launch --nproc_per_node=4 --master_port 50003 main.py
if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="config/ped_yolov4.yaml")
    processor.run()
