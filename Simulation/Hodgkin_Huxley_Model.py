# Hodgkin-Huxley Model实现

# 导库
import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

# 构建HH模型类
class HH(bp.NeuGroup):
    def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387,
                 gL=0.03, V_th=20., C=1.0, T=6.3):

        # 初始化
        super(HH, self).__init__(size=size)

        # 定义神经元参数
        self.ENa = ENa                                              # Na离子的平衡电位（即平衡离子浓度差引起的离子定向移动所需的等效电位）
        self.EK = EK                                                # K离子的平衡电位
        self.EL = EL                                                # 泄露通道的平衡单位
        self.gNa = gNa                                              # Na离子通道的最大电导
        self.gK = gK                                                # K离子通道的最大电导
        self.gL = gL                                                # 泄露通道的最大电导
        self.C = C                                                  # 电容（表示细胞膜储存电荷的能力）
        self.V_th = V_th                                            # 膜电位阈值
        self.Q10 = 3.                                               # 温度升高10℃的速率比
        self.T_base = 6.3                                           # 基准温度
        self.phi = self.Q10 ** ((T - self.T_base) / 10)             # 温度因子，其中T是目标温度

        # 定义神经元变量
        self.V = bm.Variable(-70.68 * bm.ones(self.num))            # 膜电位
        self.m = bm.Variable(0.0266 * bm.ones(self.num))            # 离子通道门控变量m
        self.h = bm.Variable(0.772 * bm.ones(self.num))             # 离子通道门控变量h
        self.n = bm.Variable(0.235 * bm.ones(self.num))             # 离子通道门控变量n
        self.input = bm.Variable(bm.zeros(self.num))                  # 神经元接收的外部输入电流
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))    # 神经元发放状态

        # 神经元上次发放的时刻
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # 定义积分函数
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')

    # 导数
    @property                                                       # 装饰器：描述符
    def derivative(self):
        return bp.JointEq(self.dV, self.dm, self.dh, self.dn)

    # 钠离子通道亚基激活
    def dm(self, m, t, V):
        alpha = 0.1 * (V + 40) / (1 - (bm.exp(-(V + 40) / 10)))     # 阀门从开到关的速率
        beta = 4.0 * (bm.exp(-(V + 65) / 18))                       # 阀门从关到开的速率
        dmdt = alpha * (1 - m) - beta * m
        return self.phi * dmdt

    # 钠离子通道亚基失活
    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-((V + 65) / 20.))                     # 阀门从开到关的速率
        beta = 1 / (1 + bm.exp(-(V + 35) / 10))                     # 阀门从关到开的速率
        dhdt = alpha * (1 - h) - beta * h
        return self.phi * dhdt

    # 钾离子通道亚基激活
    def dn(self, n, t, V):
        alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))      # 阀门从开到关的速率
        beta = 0.125 * bm.exp(-((V + 65) / 80))                     # 阀门从关到开的速率
        dndt = alpha * (1 - n) - beta * n
        return self.phi * dndt

    # 膜电位
    def dV(self, V, t, m, h, n):
        I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
        I_K = (self.gK * n ** 4.0) * (V - self.EK)
        I_Leak = self.gL * (V - self.EL)
        dVdt = (- I_Na - I_K - I_Leak + self.input) / self.C
        return dVdt

    # 更新参数
    def update(self, tdi):
        t, dt = tdi.t, tdi.dt

        # 更新下一时刻的变量值
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, t, dt=dt)

        # 判断神经元是否产生动作电位：利用逻辑与，使当且仅当前一时刻膜电位在阈值之下而下一时刻膜电位到达阈值时，产生脉冲
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)

        # 更新神经元发放的时刻
        self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)

        # 变量替换
        self.V.value = V
        self.m.value = m
        self.h.value = h
        self.n.value = n
        self.input[:] = 0.                                            #  重置神经元接收到的外部输入电流


# 利用brainpy.inputs.section_input()函数设置时长均为2ms，但大小不同的外部输入电流
currents, length = bp.inputs.section_input(
    values=[0., bm.asarray([1., 2., 4., 8., 10., 15.]), 0.],        # 不同的电流大小
    durations=[10, 2, 25],                                          # 持续时间
    return_length=True)

# 实例化
hh = HH(currents.shape[1])
runner = bp.DSRunner(hh,
                    monitors=['V', 'm', 'h', 'n'],
                    inputs=['input', currents, 'iter'])
runner.run(length)

# 可视化
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='Volatge(mV)',
                       plot_ids=np.arange(currents.shape[1]), )

# 将外部输入电流的变化画在膜电位变化的下方
plt.plot(runner.mon.ts, bm.where(currents[:, -1] > 0, 10., 0.).numpy() - 90)
plt.tight_layout()
plt.show()


