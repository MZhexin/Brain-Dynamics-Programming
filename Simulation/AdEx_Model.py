# 适应性指数整合发放模型（AdEx Model）

# 导库
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

# 构建AdEx类
class AdEx(bp.dyn.NeuDyn):
    def __init__(self, size, V_rest=-65., V_reset=-68., V_th=10., V_T=-60,
                 delta_T=1., a=1., b=2.5, R=1., tau=10., tau_w=30., tau_ref=0., name=None):
        # 初始化父类
        super(AdEx, self).__init__(size=size, name=name)

        # 初始化参数
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_T = V_T
        self.delta_T = delta_T
        self.a = a
        self.b = b
        self.R = R
        self.tau = tau
        self.tau_w = tau_w
        self.tau_ref = tau_ref

        # 初始化变量
        self.V = bm.Variable(bm.random.randn(self.num) + V_rest)
        self.w = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

        # 定义积分器
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')

    def dV(self, V, t, w, Iext):
        tmp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dVdt = (-V + self.V_rest + tmp - self.R * w + self.R * Iext) / self.tau
        return dVdt

    def dw(self, w, t, V):
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        return dwdt

    # 将两个微分方程联合为1个，以便同时积分
    @property
    def derivative(self):
        return bp.JointEq(self.dV, self.dw)

    def update(self):
        t, dt = bp.share['t'], bp.share['dt']
        V, w = self.integral(self.V, self.w, t, self.input, dt=dt)
        refractory = (t - self.t_last_spike) <= self.tau_ref
        V = bm.where(refractory, self.V, V)
        spike = V > self.V_th
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.w.value = bm.where(spike, w + self.b, w)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.input[:] = 0.

# 运行AdEx模型
neu = AdEx(1)
runner = bp.DSRunner(neu, monitors=['V', 'w', 'spike'], inputs=('input', 9.), dt=0.01)
runner(500)

# 可视化
runner.mon.V = bm.where(runner.mon.spike, 20., runner.mon.V)
plt.plot(runner.mon.ts, runner.mon.V, label='V')
plt.plot(runner.mon.ts, runner.mon.w, label='w')
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.show()
