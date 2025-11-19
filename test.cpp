#include "GPB.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

//-----------------------------------------------
// 自适应 Crank-Nicolson (CN) 积分器
//-----------------------------------------------
int main()
{
    std::cout << "Starting implicit CN simulation for GPB single cell...\n";

    const double t_end = 1000.0;      // ms
    double dt = 0.1;                 // 初始时间步 (ms)
    const double dt_min = 1e-5;       // 最小步长
    const double dt_max = 0.1;        // 最大步长
    const int subdomain_id = 1;       // 细胞类型标识

    std::vector<double> y(GPBCellModel::ODE_DIM, 0.0);
    double y0[58] = {0.00600136385183703,0.543601084954698,0.527803785787066,4.23728460444658e-06,0.993619374470099,0.0505387235922671,0.0210017777070521,0.000522877195100256,0.832186276845895,0.000522776733509404,0.999993508389616,0.0390721892742666,0.00500094825839587,0.834614642478600,3.75319906663631e-06,7.43705247074737e-07,3.64691611958351,0.795075283757468,0.0119727572903327,0.123958774552743,0.00750602540301977,0.000410732235396523,0.00289976879177577,0.136576139054802,0.00289523367635249,0.0152843675893692,0.0135261758550217,0.109758215767868,0.143268523727843,1.21836783167243,0.573190684765844,9.32956858749209,9.31201948078332,9.31173701160579,120,0.000368296788745360,0.000146267082986440,0.000121716170407605,-79.2265229989188,0.994600000000000,1,0.00400449827286187,0.0988773961817519,0.149400000000000,0.407100000000000,0.416100000000000,-0.314529004000000,0.000100000000000000,0.000600000000000000,0.000800000000000000,0,0,0,0,0,0,0
    };
    for (int i = 0; i < GPBCellModel::ODE_DIM; ++i)
        y[i] = y0[i];

    double Vm = y[38];
    double t = 0.0;

    // ------------------------------
    // 输出设置
    // ------------------------------
    std::ofstream fout("gpb_CN_output.csv");
    fout << "time_ms,Vm_mV\n";
    fout << std::fixed << std::setprecision(6);
    fout << t << "," << Vm << "\n";

    // ------------------------------
    // 时间积分主循环
    // ------------------------------
    while (t < t_end)
    {
        double I_app = ((t >= 0.0 && t <= 2.0) /*|| (t >= 310.0 && t <= 315.0)*/) ? 9.5 : 0.0;

        // predictor: 显式Euler预测
        auto res_n = GPBCellModel::f_heartfailure_1(t, y, y[38], I_app);
        std::vector<double> y_pred(y.size());
        for (size_t i = 0; i < y.size(); ++i)
            y_pred[i] = y[i] + dt * res_n.ydot[i];

        // corrector: CN迭代 (固定点或简化Newton)
        std::vector<double> y_new = y_pred;
        bool converged = false;

        for (int iter = 0; iter < 20; ++iter)
        {
            auto res_np1 = GPBCellModel::f_heartfailure_1(t + dt, y_new, y_new[38], I_app);
            std::vector<double> y_corr(y.size());
            double err = 0.0;

            for (size_t i = 0; i < y.size(); ++i)
            {
                y_corr[i] = y[i] + 0.5 * dt * (res_n.ydot[i] + res_np1.ydot[i]);
                err += std::fabs(y_corr[i] - y_new[i]);
            }

            err /= y.size();
            y_new = y_corr;

            if (err < 1e-7) { converged = true; break; }
        }

        if (!converged)
        {
            dt *= 0.1;  // 收敛失败 → 缩小步长
            if (dt < dt_min) {
                std::cerr << "Time step below minimum at t=" << t << " ms\n";
                break;
            }
            continue;
        }

        // 更新状态
        y = y_new;
        Vm = y[38];
        t += dt;

        fout << t << "," << Vm << "\n";

        // 自适应时间步调整
        dt = std::max(dt_min, std::min(dt * 1.2, dt_max));


        if (std::isnan(Vm) || std::fabs(Vm) > 200.0)
        {
            std::cerr << "Divergence detected at t=" << t << " Vm=" << Vm << "\n";
            break;
        }

        if (std::fmod(t, 10.0) < dt)
            std::cout << "t=" << t << " ms, Vm=" << Vm << " mV, dt=" << dt << " ms\n";
    }

    fout.close();
    std::cout << "Simulation complete, results in gpb_CN_output.csv\n";
    return 0;
}
