# LIF模型

# 导库
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

# 定义LIF类
class LIF(bp.dyn.NeuDyn):
    def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., R=1., tau=10., t_ref=5., **kwargs):
        # 初始化父类
        super(LIF, self).__init__(size=size, **kwargs)

        # 初始化参数
        self.V_rest = V_rest            # 静息电位
        self.V_reset = V_reset          # 重置电位
        self.V_th = V_th                # 阈值
        self.R = R                      # 电阻
        self.tau = tau                  # 时间常数：表示V随时间衰减的速率
        self.t_ref = t_ref              # 不应期时长

        # 初始化变量
        self.V = bm.Variable(bm.random.randn(self.num) + V_rest)            # 膜电位，bm.random.randn()制造随机噪音
        self.input = bm.Variable(bm.zeros(self.num))                        # 外部输入电流
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)           # 上一次发放的时间，处理不应期时使用
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))       # 是否处于不应期
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))            # 脉冲发放状态

        # 使用指数欧拉方法进行积分
        self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

    # 定义膜电位关于时间变化的微分方程
    def derivative(self, V, t, Iext):
        dVdt = (-V + self.V_rest + self.R * Iext) / self.tau
        return dVdt

    def update(self):
        t, dt = bp.share['t'], bp.share['dt']
        # 以数组方式对神经元进行更新
        refractory = (t - self.t_last_spike) <= self.t_ref                  # 判断神经元是否处于不定期
        V = self.integral(self.V, t, self.input, dt=dt)                     # 根据时间步长更新膜电位
        V = bm.where(refractory, self.V, V)                                 # 若处于不定期，发放原始膜电位self.V，否则返回更新后的膜电位V
        spike = V > self.V_th                                               # 将大于阈值的神经元标记为发放状态
        self.spike[:] = spike                                               # 更新神经元脉冲发放状态
        self.t_last_spike[:] = bm.where(spike, t, self.t_last_spike)        # 更新神经元最后一次脉冲的时间
        self.V[:] = bm.where(spike, self.V_reset, V)                        # 将发放脉冲的神经元的膜电位置为V_reset，其余不变
        self.refractory[:] = bm.logical_or(refractory, spike)               # 更新神经元是否处于不定期
        self.input[:] = 0.                                                  # 重置外部输入

# 运行LIF模型
group = LIF(1)
runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 22.))
runner(200)     # 运行时间为200ms

# 结果可视化
fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
ax = fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon.V)
plt.xlabel(r't$ (ms)')
plt.ylabel(r'$V$ (mV)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
