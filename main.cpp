#include <iostream>
#include <fstream>
#include <dolfin.h>
#include "EMcomping.h"
#include "GPBTissueManager.h"

namespace dolfin {

// 纤维方向 (z方向)
class FiberDirection : public Expression
{
public:
    FiberDirection() : Expression(3) {}
    void eval(Eigen::Ref<Eigen::VectorXd> values,
              Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override
    {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 1.0;
    }
};

// 片层方向 (x方向)
class SheetDirection : public Expression
{
public:
    SheetDirection() : Expression(3) {}
    void eval(Eigen::Ref<Eigen::VectorXd> values,
              Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override
    {
        values[0] = 1.0;
        values[1] = 0.0;
        values[2] = 0.0;
    }
};

// 刺激 (z≈0 处, 时间依赖)
class StimulusCurrent : public Expression
{
public:
    StimulusCurrent(double amplitude, double t_start, double t_end)
        : Expression(), amp(amplitude), t_start(t_start), t_end(t_end), current_time(0.0) {}
    
    void eval(dolfin::Array<double>& values,
              const dolfin::Array<double>& x) const override
    {
        // 在 z ≈ 0 且 时间在 [t_start, t_end] 内施加刺激
        if (std::fabs(x[2]) < 0.05 && 
            current_time >= t_start && current_time <= t_end) {
            values[0] = amp;  // μA/cm²
        } else {
            values[0] = 0.0;
        }
    }
    
    void set_time(double t) { current_time = t; }

private:
    double amp, t_start, t_end;
    mutable double current_time;
};

} // namespace dolfin

// ... (keep all your includes and class definitions the same)

int main()
{
    try {
        std::cout << "\n╔═══════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  单域方程 + GPB 细胞模型                      ║" << std::endl;
        std::cout << "║  算子分裂法 + 子循环 (Subcycling)            ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════╝\n" << std::endl;
        
        // ════════════════════════════════════════════════════════
        // 1. Mesh
        // ════════════════════════════════════════════════════════
        auto mesh = std::make_shared<dolfin::Mesh>(
            dolfin::BoxMesh(
                dolfin::Point(0.0, 0.0, 0.0),
                dolfin::Point(0.25, 0.25, 0.75),
                8, 8, 24
            )
        );
        std::cout << "网格信息:" << std::endl;
        std::cout << "  顶点数: " << mesh->num_vertices() << std::endl;
        std::cout << "  单元数: " << mesh->num_cells() << std::endl;
        
        // ════════════════════════════════════════════════════════
        // 2. 多尺度时间参数设置
        // ════════════════════════════════════════════════════════
        
        // PDE time step (tissue scale) - LARGER (coarse time scale)
        double dt_pde_ms = 0.005;                     // 50 microseconds = 0.05 ms
        double dt_pde_seconds = dt_pde_ms / 1000.0;  // Convert to seconds
        
        // ODE time step (cell scale) - SMALLER
        double dt_ode_ms = 0.001;                     // 50 microseconds = 0.05 ms
        
        // Total simulation time
        double T_total_seconds = 1.0;                // 1000 ms total
        
        // Calculate number of PDE steps (not ODE steps!)
        size_t num_pde_steps = static_cast<size_t>(T_total_seconds / dt_pde_seconds);
        
        // Calculate subcycling ratio
        int subcycling_ratio = static_cast<int>(std::round(dt_pde_ms / dt_ode_ms));
        
        std::cout << "\n多尺度时间参数:" << std::endl;
        std::cout << "  ┌─────────────────────────────────────────┐" << std::endl;
        std::cout << "  │ PDE (组织) 时间步长: " << dt_pde_ms << " ms     │" << std::endl;
        std::cout << "  │ ODE (细胞) 时间步长: " << dt_ode_ms << " ms     │" << std::endl;
        std::cout << "  │ 子循环比率:          " << subcycling_ratio << "           │" << std::endl;
        std::cout << "  └─────────────────────────────────────────┘" << std::endl;
        std::cout << "  总时间: " << T_total_seconds << " s = " 
                  << T_total_seconds * 1000 << " ms" << std::endl;
        std::cout << "  PDE 总步数: " << num_pde_steps << std::endl;
        std::cout << "  理论加速比: " << subcycling_ratio << "x" << std::endl;
        
        // ════════════════════════════════════════════════════════
        // 3. Function spaces and functions
        // ════════════════════════════════════════════════════════
        auto V_scalar = std::make_shared<EMcomping::FunctionSpace>(mesh);
        auto V_vector = std::make_shared<EMcomping::Form_a_FunctionSpace_2>(mesh);
        
        std::cout << "\n函数空间:" << std::endl;
        std::cout << "  自由度 (DOFs): " << V_scalar->dim() << std::endl;
        
        auto Vm = std::make_shared<dolfin::Function>(V_scalar);
        auto Vm_old = std::make_shared<dolfin::Function>(V_scalar);
        auto I_ion = std::make_shared<dolfin::Function>(V_scalar);
        auto I_stim = std::make_shared<dolfin::Function>(V_scalar);
        
        // Initial condition: resting potential
        dolfin::Constant V_rest(-0.085);  // -85 mV
        Vm->interpolate(V_rest);
        Vm_old->interpolate(V_rest);
        
        std::cout << "  初始 Vm: " << Vm->vector()->max() * 1000 << " mV" << std::endl;
        
        // ════════════════════════════════════════════════════════
        // 4. Fiber and sheet directions
        // ════════════════════════════════════════════════════════
        auto fiber_func = std::make_shared<dolfin::Function>(V_vector);
        auto sheet_func = std::make_shared<dolfin::Function>(V_vector);
        
        auto f0_expr = std::make_shared<dolfin::FiberDirection>();
        auto s0_expr = std::make_shared<dolfin::SheetDirection>();
        
        fiber_func->interpolate(*f0_expr);
        sheet_func->interpolate(*s0_expr);
        
        std::cout << "\n纤维方向: (0, 0, 1)" << std::endl;
        std::cout << "片层方向: (1, 0, 0)" << std::endl;
        
        // ════════════════════════════════════════════════════════
        // 5. Stimulus setup
        // ════════════════════════════════════════════════════════
        double stim_amplitude = 5.0;      // μA/cm²
        double stim_start = 0.0;          // s
        double stim_end = 2.0e-3;         // 2 ms
        
        auto stim_expr = std::make_shared<dolfin::StimulusCurrent>(
            stim_amplitude, stim_start, stim_end
        );
        
        std::cout << "\n刺激参数:" << std::endl;
        std::cout << "  幅值: " << stim_amplitude << " μA/cm²" << std::endl;
        std::cout << "  持续时间: " << stim_start << " - " << stim_end * 1000 << " ms" << std::endl;
        std::cout << "  位置: z ≈ 0" << std::endl;
        
        // ════════════════════════════════════════════════════════
        // 6. Initialize GPB tissue manager WITH SUBCYCLING
        // ════════════════════════════════════════════════════════
        std::cout << std::endl;
        auto gpb_tissue = std::make_shared<GPBTissueManager>(
            V_scalar, 
            dt_pde_ms,   // PDE time step
            dt_ode_ms    // ODE time step
        );
        
        // ════════════════════════════════════════════════════════
        // 7. Setup PDE solver
        // ════════════════════════════════════════════════════════
        std::cout << "\n设置单域方程 PDE 求解器..." << std::endl;
        
        auto a = std::make_shared<EMcomping::Form_a>(V_scalar, V_scalar);
        auto L = std::make_shared<EMcomping::Form_L>(V_scalar);
        
        // IMPORTANT: Use PDE time step here!
        a->f0 = fiber_func;
        a->s0 = sheet_func;
        a->dt = std::make_shared<dolfin::Constant>(dt_pde_seconds);
        
        L->Vm_old = Vm_old;
        L->I_ion = I_ion;
        L->I_stim = I_stim;
        L->dt = std::make_shared<dolfin::Constant>(dt_pde_seconds);
        
        std::vector<std::shared_ptr<const dolfin::DirichletBC>> bcs;
        
        auto pde_problem = std::make_shared<dolfin::LinearVariationalProblem>(
            a, L, Vm, bcs
        );
        
        auto pde_solver = std::make_shared<dolfin::LinearVariationalSolver>(pde_problem);
        pde_solver->parameters["linear_solver"] = "lu";
        pde_solver->parameters["preconditioner"] = "default";
        
        std::cout << "  求解器: LU 分解 (直接法)" << std::endl;
        std::cout << "  PDE 时间步长: " << dt_pde_seconds << " s" << std::endl;
        
        // ════════════════════════════════════════════════════════
        // 8. Output setup
        // ════════════════════════════════════════════════════════
        dolfin::File vm_file("results/Vm.pvd");
        std::ofstream csv_file("results/statistics.csv");
        csv_file << "time_ms,Vm_min_mV,Vm_max_mV,Vm_mean_mV,"
                 << "I_ion_min,I_ion_max,I_ion_mean\n";
        
        // ════════════════════════════════════════════════════════
        // 9. TIME STEPPING LOOP (PDE steps, not ODE steps!)
        // ════════════════════════════════════════════════════════
        std::cout << "\n╔═══════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  开始时间步进 (多尺度方法)                   ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════╝\n" << std::endl;
        
        size_t output_interval = 200;  // Output every 20 PDE steps
        double t_seconds = 0.0;
        
        for (size_t pde_step = 0; pde_step < num_pde_steps; ++pde_step)
        {
            double t_milliseconds = t_seconds * 1000.0;
            
            // ┌──────────────────────────────────────────────┐
            // │ Operator Splitting Step 1: ODE with SUBCYCLING
            // │ Input:  Vm^n (held constant)
            // │ Output: I_ion^n (time-averaged over substeps)
            // │ Effect: Cell states advance by dt_pde_ms
            // └──────────────────────────────────────────────┘
            gpb_tissue->compute_ionic_current_subcycling(
                Vm_old, 
                I_ion, 
                t_milliseconds
            );
            
            // ┌──────────────────────────────────────────────┐
            // │ Update stimulus
            // └──────────────────────────────────────────────┘
            stim_expr->set_time(t_seconds);
            I_stim->interpolate(*stim_expr);
            
            // ┌──────────────────────────────────────────────┐
            // │ Operator Splitting Step 2: PDE
            // │ Input:  Vm^n, I_ion^n (averaged), I_stim^n
            // │ Output: Vm^{n+1}
            // └──────────────────────────────────────────────┘
            pde_solver->solve();
            
            // ┌──────────────────────────────────────────────┐
            // │ Check numerical stability
            // └──────────────────────────────────────────────┘
            if (!gpb_tissue->check_stability(Vm, I_ion, 200.0)) {
                std::cerr << "\n在 PDE 步骤 " << pde_step << " 检测到数值不稳定!" << std::endl;
                
                auto stats = gpb_tissue->get_statistics(Vm, I_ion);
                std::cerr << "  Vm: [" << stats.Vm_min_mV << ", " 
                          << stats.Vm_max_mV << "] mV" << std::endl;
                std::cerr << "  I_ion: [" << stats.I_ion_min << ", " 
                          << stats.I_ion_max << "] μA/cm²" << std::endl;
                
                throw std::runtime_error("数值不稳定");
            }
            
            // ┌──────────────────────────────────────────────┐
            // │ Update: Vm_old ← Vm
            // └──────────────────────────────────────────────┘
            *Vm_old = *Vm;
            
            // ┌──────────────────────────────────────────────┐
            // │ Output
            // └──────────────────────────────────────────────┘
            if (pde_step % output_interval == 0) {
                auto stats = gpb_tissue->get_statistics(Vm, I_ion);
                
                std::cout << "PDE 步骤 " << pde_step 
                          << " (t = " << t_milliseconds << " ms):" << std::endl;
                std::cout << "  Vm: [" << stats.Vm_min_mV << ", " 
                          << stats.Vm_max_mV << "] mV (平均: " 
                          << stats.Vm_mean_mV << ")" << std::endl;
                std::cout << "  I_ion: [" << stats.I_ion_min << ", " 
                          << stats.I_ion_max << "] μA/cm² (平均: " 
                          << stats.I_ion_mean << ")" << std::endl;
                std::cout << "  (子循环: " << gpb_tissue->get_n_substeps() 
                          << " ODE 步 per PDE 步)" << std::endl;
                
                vm_file << *Vm;
                csv_file << t_milliseconds << ","
                         << stats.Vm_min_mV << ","
                         << stats.Vm_max_mV << ","
                         << stats.Vm_mean_mV << ","
                         << stats.I_ion_min << ","
                         << stats.I_ion_max << ","
                         << stats.I_ion_mean << "\n";
            }
            
            // Advance time by PDE step
            t_seconds += dt_pde_seconds;
        }
        
        csv_file.close();
        
        std::cout << "\n╔═══════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  模拟完成! (多尺度方法)                      ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════╝\n" << std::endl;
        
        std::cout << "性能统计:" << std::endl;
        std::cout << "  PDE 步数: " << num_pde_steps << std::endl;
        std::cout << "  每步子循环数: " << gpb_tissue->get_n_substeps() << std::endl;
        std::cout << "  等效 ODE 步数: " 
                  << num_pde_steps * gpb_tissue->get_n_substeps() << std::endl;
        std::cout << "  加速比: " << gpb_tissue->get_n_substeps() << "x" << std::endl;
        
        std::cout << "\n输出文件:" << std::endl;
        std::cout << "  results/Vm.pvd - ParaView 可视化" << std::endl;
        std::cout << "  results/statistics.csv - 统计数据" << std::endl;
        
        return 0;
        
    } catch (std::exception& e) {
        std::cerr << "\n╔═══════════════════════════════════════════════╗" << std::endl;
        std::cerr << "║  错误!                                         ║" << std::endl;
        std::cerr << "╚═══════════════════════════════════════════════╝\n" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
}