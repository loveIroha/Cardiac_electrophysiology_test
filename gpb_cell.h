#pragma once

#include <dolfin.h>
#include <vector>
#include <memory>
#include <cmath>
#include "GPB_cell_model.h"

using namespace dolfin;

class gpb_cell : public Expression
{
public:
    gpb_cell(std::shared_ptr<Function> I_tot_func,
             std::shared_ptr<Function> Ca_i_func,
             std::shared_ptr<const Function> Vm_func,
             double dt_sec)
        : Expression(2),
          I_tot_func(I_tot_func),
          Ca_i_func(Ca_i_func),
          Vm_func(Vm_func),
          dt(dt_sec)
    {
        V = Vm_func->function_space();
        dofmap = V->dofmap();

        // 初始化状态变量 y_i（每个节点）
        std::size_t ndofs = dofmap->global_dimension();
        std::vector<double> y0 = {
            0.00371917311175298, 0.629563145653617, 0.628156366619018, 2.87807501948292e-06, 0.995191678834277,
            0.0256455386561491, 0.0166702462776996, 0.000437233937773851, 0.789865222262672, 0.000437226960888769,
            0.999995922401368, 0.0182692097720730, 0.00425180082489623, 0.911423257257461, 1.10337211796779e-06,
            1.07229840398973e-07, 3.53795718160598, 0.772022201763961, 0.0106472771801263, 0.124747660339223,
            0.00709016718964759, 0.000361565169272969, 0.00291593396950060, 0.136552392811308, 0.00259137579210590,
            0.00770065038544987, 0.0108451851114268, 0.0754357237706355, 0.122427602372974, 1.33525927996781,
            0.686242333225035, 8.78601782663918, 8.78614841263188, 8.78640538366833, 120.0, 0.000182997516145308,
            0.000117027510847066, 0.000107032812487738, -81.5455936324844, 0.9946, 1.0, 0.00273690302369662,
            0.168523952444531, 0.1494, 0.4071, 0.4161, -0.314529004, 0.0001, 0.0006, 0.0008,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        }; // 共 58 个值

        y_list.resize(ndofs, y0); // 所有节点初始状态一致地赋为 y0

    }

    // GPB节点更新器（每一步调用一次）
    void update_from_Vm(double t)
    {
        std::vector<double> Vm_vec;
        Vm_func->vector()->get_local(Vm_vec);  // 把所有 dof 上的 Vm 存入 Vm_vec[i]

        std::size_t ndofs = Vm_vec.size();

        std::vector<double> I_vec(ndofs, 0.0);
        std::vector<double> Ca_vec(ndofs,0.0);

        for (std::size_t i = 0; i < ndofs; ++i)
        {
            double Vm_local = Vm_vec[i] * 1000 ; // V 单位 mV
            int subdomain_id = 1;
            auto& y = y_list[i];
            auto res = GPBCellModel::f_heartfailure_1(t, y, Vm_local, subdomain_id,0);
            const auto& ydot = res.ydot;

            for (std::size_t j = 0; j < 12; ++j){
                if (j == 5 || j == 6) 
                {y[j] += dt * ydot[j];
                continue;
                }  // fCa_junc / fCa_sl 保持欧拉
                double y_inf = 0.0, tau = 1.0;
                GPBCellModel::gating_inf_tau((int)j, Vm_local, y_inf, tau);
                // y_new = y_inf - (y_inf - y_old) * exp(-dt/tau)
                if (tau < 1e-10)
                    y[j] = y_inf + (y[j] - y_inf) * std::exp(-dt / tau);
                else
                    y[j] = y_inf;
            }
                // Step 2: 其余变量欧拉；Vm (38) 由 PDE 管
            for (std::size_t j = 13; j < y.size(); ++j){
                if (j == 38) continue;      // Vm 交给 PDE
                y[j] += dt * ydot[j];       // 显式欧拉
            }

            I_vec[i] = res.I_tot ;  // I_tot, µA/µF → µA/cm²
            Ca_vec[i] = y[37];    // Ca_i
        }

        I_tot_func->vector()->set_local(I_vec);
        I_tot_func->vector()->apply("insert");
        Ca_i_func->vector()->set_local(Ca_vec);
        Ca_i_func->vector()->apply("insert");

        // 更新缓存向量用于 eval
        I_tot_vec = std::move(I_vec);
        Ca_i_vec = std::move(Ca_vec);
    }

    // FEniCS在评估表达式时调用
    void eval(Eigen::Ref<Eigen::VectorXd> values,
              Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell &cell) const override
    {
        dolfin::Cell dolfin_cell(*V->mesh(), cell.index);
        auto dofs = dofmap->cell_dofs(dolfin_cell.index());

        std::size_t closest_dof = dofs[0];
        double min_dist = std::numeric_limits<double>::max();

        const auto& coords = V->tabulate_dof_coordinates();
        std::size_t dim = V->mesh()->geometry().dim();

        for (std::size_t i = 0; i < dofs.size(); ++i)
        {
            std::size_t dof = dofs[i];
            double dist = 0.0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                double dx = x[d] - coords[dof * dim + d];
                dist += dx * dx;
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_dof = dof;
            }
        }

        values[0] = I_tot_vec[closest_dof];
        values[1] = Ca_i_vec[closest_dof];
    }

public:
    std::shared_ptr<Function> I_tot_func;
    std::shared_ptr<Function> Ca_i_func;
    std::shared_ptr<const Function> Vm_func;

    std::shared_ptr<const FunctionSpace> V;
    std::shared_ptr<const GenericDofMap> dofmap;

    std::vector<std::vector<double>> y_list; // 每个节点的 GPB 状态变量
    std::vector<double> I_tot_vec;
    std::vector<double> Ca_i_vec;

    double dt;
};
