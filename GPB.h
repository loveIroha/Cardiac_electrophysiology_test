#pragma once
#include <vector>
#include <cmath>

class GPBCellModel
{
public:

    struct Result {
        std::vector<double> ydot; // ODE 的状态导数
        double I_tot;             // 总跨膜电流
    };

    static constexpr int ODE_DIM = 58; // 状态变量数
    // static void gating_inf_tau(int j, double Vm, double& y_inf, double& tau);
    // 计算 ODE 右端项
    static Result f_heartfailure_1(
        double t,
        std::vector<double>& y,
        double Vm, double I_app
    );

};
