import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Topology:
    def __init__(self, kind, size, link_type, alpha):
        self.kind = kind
        self.size = size
        self.link_type = link_type
        self.alpha = alpha # [ ms]

    def get_bw(self) -> float:
        bw = {
            "infiniband" : 0.8,
            "PCIe" : 0.256,
            "nvlink" : 1.8,
            "l2" : 6,
            "HBM" : 8,
            "HBI" : 10,
            "l1" : 20
        } # [TB/s]
        return bw[self.link_type]

    def get_t_comm(self, v, c):
        log_size = np.ceil(np.log2(self.size))
        t1 = self.alpha * log_size
        t2 = (v * log_size * 2) / (c * self.get_bw())
        return t1 + t2


class GPU:
    def __init__(self, efficiency_factor, t_launch):
        self.efficiency_factor = efficiency_factor
        self.t_launch = t_launch # [ms]
        self.tensor_memory = 256e3 #[Bytes]

    def get_achieved_tflops(self, element_type) -> float:
        peak_tflops = {
            "fp32" : 495,
            "fp16" : 990,
            "bf16" : 990,
            "fp8" : 1980,
            "nvfp4" : 15000
        } # [Tera Floating Point Ops per Second]
        achieved_tflops = peak_tflops[element_type] * self.efficiency_factor
        return achieved_tflops


class Simulator:
    def __init__(self, inner_topology, outer_topology, gpu, dim, r, element_type):
        self.inner_topology = inner_topology
        self.outer_topology = outer_topology
        self.gpu = gpu
        self.dim = dim
        self.r = r
        self.element_type = element_type

    def get_elem_size(self) -> int:
        element_size = {
            "fp32" : 4,
            "fp16" : 2,
            "bf16" : 2,
            "fp8" : 1,
            "nvfp4" : 0.5
        }
        return element_size[self.element_type]

    def calculate_t_comp(self, b, c) -> float:
        total_gpus = self.inner_topology.size * self.outer_topology.size
        flops = 2 * (b / c) * self.dim * (self.r * self.dim / total_gpus) # make sure this is Tera-FLOPS
        achieved_flops = self.gpu.get_achieved_tflops(self.element_type)
        return self.gpu.t_launch + (flops / achieved_flops)

    def calculate_t_comm(self, b, c) -> float:
        element_size = self.get_elem_size()
        num_gpus = self.inner_topology.size * self.outer_topology.size
        v_gpu = (1 / num_gpus) * b * self.r * self.dim * element_size
        v_outer_top = (1 / self.outer_topology.size) * b * self.r * self.dim * element_size
        t_inner_top = self.inner_topology.get_t_comm(v_gpu, c)
        t_outer_top = self.outer_topology.get_t_comm(v_outer_top, c)
        return t_outer_top + t_inner_top

    def opt_func_1(self, param) -> float:
        b = param[0]
        c = param[1]
        t_comp_a = t_comp_b = self.calculate_t_comp(b, c)
        t_comm = self.calculate_t_comm(b, c)
        t_total = t_comp_a + np.maximum(t_comp_b, t_comm)
        return t_total

    def opt_func_2(self, param) -> float:
        b = param[0]
        c = param[1]
        t_comp_a = t_comp_b = self.calculate_t_comp(b, c)
        t_comm = self.calculate_t_comm(b, c)
        t_total = t_comp_a + np.maximum(t_comp_b, t_comm)
        return - (b / t_total)

    def optimize(self):
        weights = np.linspace(0, 1, 10)
        x0 = np.array([2, 2])
        b_opt = []
        c_opt = []
        for w in weights:
            objective = lambda p: w * self.opt_func_1(p) + (1 - w) * self.opt_func_2(p)
            bounds = [(1, self.gpu.tensor_memory), (1, 9)]
            res = minimize(objective, x0, bounds=bounds, method="Nelder-Mead")
            if res.success:
                b_opt.append(res.x[0])
                c_opt.append(res.x[1])
            else:
                print("Optimization Failed")
        plt.figure()
        plt.scatter(b_opt, c_opt)
        plt.xlabel('B')
        plt.ylabel('C')
        plt.show()


nvlink_topology = Topology("something", 72, "nvlink", 1)
infiniband_topology = Topology("something", 2, "infiniband", 3)
gpu = GPU(0.4, 1)
sim = Simulator(nvlink_topology, infiniband_topology, gpu, 4096, 4, "fp32")
sim.optimize()

