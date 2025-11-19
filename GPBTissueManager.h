#pragma once
#include <dolfin.h>
#include "GPB_cell_model.h"
#include <vector>
#include <memory>

/**
 * GPBTissueManager - Multi-scale time stepping with subcycling
 * 
 * Time scale separation strategy:
 * - PDE (tissue): Large time step (dt_PDE ~ 0.05-0.1 ms)
 * - ODE (cell):   Small time steps (dt_ODE ~ 0.01 ms)
 * - Subcycling:   Multiple ODE steps per PDE step
 * 
 * Operator splitting with subcycling:
 *   For each PDE step Δt_PDE:
 *     1. Fix Vm at current value
 *     2. Take N_sub ODE substeps: dt_ODE = Δt_PDE / N_sub
 *     3. Average I_ion over substeps
 *     4. Use averaged I_ion for PDE step to get Vm^{n+1}
 */
class GPBTissueManager
{
public:
    /**
     * Constructor with subcycling support
     * 
     * @param function_space  FEniCS scalar function space
     * @param dt_pde_ms      PDE time step (milliseconds) - larger
     * @param dt_ode_ms      ODE time step (milliseconds) - smaller
     */
    GPBTissueManager(
        std::shared_ptr<dolfin::FunctionSpace> function_space, 
        double dt_pde_ms,
        double dt_ode_ms)
        : V_space(function_space), 
          dt_pde_milliseconds(dt_pde_ms),
          dt_ode_milliseconds(dt_ode_ms)
    {
        // Calculate number of ODE substeps per PDE step
        n_substeps = static_cast<int>(std::round(dt_pde_ms / dt_ode_ms));
        
        // Ensure at least 1 substep
        if (n_substeps < 1) {
            std::cerr << "Warning: dt_ODE > dt_PDE, setting n_substeps = 1" << std::endl;
            n_substeps = 1;
            dt_ode_milliseconds = dt_pde_milliseconds;
        }
        
        // Adjust dt_ode to exactly divide dt_pde
        dt_ode_milliseconds = dt_pde_milliseconds / n_substeps;
        
        num_nodes = V_space->dim();
        
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cout << "初始化 GPB 组织管理器 (多尺度时间步进)" << std::endl;
        std::cout << "  网格节点数: " << num_nodes << std::endl;
        std::cout << "  每节点状态数: " << GPBCellModel::CELL_STATE_DIM << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cout << "时间尺度分离策略:" << std::endl;
        std::cout << "  PDE 时间步长 (Δt_tissue): " << dt_pde_milliseconds << " ms" << std::endl;
        std::cout << "  ODE 时间步长 (Δt_cell):   " << dt_ode_milliseconds << " ms" << std::endl;
        std::cout << "  子循环步数 (N_substeps):  " << n_substeps << std::endl;
        std::cout << "  加速比 (理论):            " << n_substeps << "x" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        
        // Initialize cell states
        cell_states.resize(num_nodes);
        std::vector<double> resting_state = GPBCellModel::get_initial_state();
        
        for (size_t i = 0; i < num_nodes; ++i) {
            cell_states[i] = resting_state;
        }
        
        std::cout << "  所有细胞已初始化为静息态" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    }
    
    /**
     * Compute ionic current with subcycling
     * 
     * Algorithm:
     * 1. Hold Vm constant at current value
     * 2. Take n_substeps ODE substeps with dt_ode
     * 3. Average I_ion over all substeps
     * 4. Return averaged I_ion for PDE
     * 
     * @param Vm_function     Current membrane potential (V)
     * @param I_ion_function  Output averaged ionic current (μA/cm²)
     * @param time_ms         Current time (milliseconds)
     */
    void compute_ionic_current_subcycling(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double time_ms)
    {
        // Get Vm array from FEniCS
        std::vector<double> Vm_array;
        Vm_function->vector()->get_local(Vm_array);
        
        if (Vm_array.size() != num_nodes) {
            throw std::runtime_error("Vm array size mismatch!");
        }
        
        // Prepare output array for averaged I_ion
        std::vector<double> I_ion_avg(num_nodes, 0.0);
        
        // ═══════════════════════════════════════════════════
        // SUBCYCLING LOOP: Take multiple ODE substeps
        // ═══════════════════════════════════════════════════
        for (int substep = 0; substep < n_substeps; ++substep) {
            
            double t_sub = time_ms + substep * dt_ode_milliseconds;
            
            // Process all nodes at this substep
            for (size_t node = 0; node < num_nodes; ++node) {
                
                // Convert V → mV
                double Vm_volts = Vm_array[node];
                double Vm_millivolts = Vm_volts * 1000.0;
                
                // Call GPB cell model
                GPBCellModel::Result result = GPBCellModel::compute_ionic_current(
                    t_sub,
                    cell_states[node],
                    Vm_millivolts
                );
                
                // Check validity
                if (std::isnan(result.I_ion) || std::isinf(result.I_ion)) {
                    std::cerr << "Warning: Node " << node 
                              << " substep " << substep 
                              << " invalid I_ion = " << result.I_ion << std::endl;
                    result.I_ion = 0.0;
                }
                
                // Accumulate I_ion for averaging
                I_ion_avg[node] += result.I_ion;
                
                // Update cell states using Forward Euler
                for (size_t j = 0; j < GPBCellModel::CELL_STATE_DIM; ++j) {
                    cell_states[node][j] += dt_ode_milliseconds * result.ydot[j];
                    
                    // Check state validity
                    if (std::isnan(cell_states[node][j]) || std::isinf(cell_states[node][j])) {
                        std::cerr << "Warning: Node " << node 
                                  << " state " << j 
                                  << " became NaN/Inf, resetting to 0" << std::endl;
                        cell_states[node][j] = 0.0;
                    }
                }
            }
        }
        
        // ═══════════════════════════════════════════════════
        // Average I_ion over all substeps
        // ═══════════════════════════════════════════════════
        for (size_t node = 0; node < num_nodes; ++node) {
            I_ion_avg[node] /= n_substeps;
        }
        
        // Write averaged I_ion back to FEniCS
        I_ion_function->vector()->set_local(I_ion_avg);
        I_ion_function->vector()->apply("insert");
    }
    
    /**
     * Original single-step method (for backward compatibility)
     */
    void compute_ionic_current(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double time_ms)
    {
        // Just call subcycling version
        compute_ionic_current_subcycling(Vm_function, I_ion_function, time_ms);
    }
    
    // Getters for time step info
    double get_dt_pde_ms() const { return dt_pde_milliseconds; }
    double get_dt_ode_ms() const { return dt_ode_milliseconds; }
    int get_n_substeps() const { return n_substeps; }
    
    /**
     * Get statistics
     */
    struct Statistics {
        double Vm_min_mV;
        double Vm_max_mV;
        double Vm_mean_mV;
        double I_ion_min;
        double I_ion_max;
        double I_ion_mean;
        size_t num_nodes;
    };
    
    Statistics get_statistics(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function) const
    {
        Statistics stats;
        stats.num_nodes = num_nodes;
        
        stats.Vm_min_mV = Vm_function->vector()->min() * 1000.0;
        stats.Vm_max_mV = Vm_function->vector()->max() * 1000.0;
        stats.Vm_mean_mV = Vm_function->vector()->sum() / num_nodes * 1000.0;
        
        stats.I_ion_min = I_ion_function->vector()->min();
        stats.I_ion_max = I_ion_function->vector()->max();
        stats.I_ion_mean = I_ion_function->vector()->sum() / num_nodes;
        
        return stats;
    }
    
    bool check_stability(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double acceptable_Vm_range_mV = 200.0) const
    {
        double Vm_min = Vm_function->vector()->min() * 1000.0;
        double Vm_max = Vm_function->vector()->max() * 1000.0;
        
        if (std::isnan(Vm_min) || std::isnan(Vm_max) ||
            std::isinf(Vm_min) || std::isinf(Vm_max)) {
            std::cerr << "Error: Vm contains NaN or Inf!" << std::endl;
            return false;
        }
        
        if (Vm_max > acceptable_Vm_range_mV || Vm_min < -acceptable_Vm_range_mV) {
            std::cerr << "Warning: Vm out of range [" << Vm_min << ", " 
                      << Vm_max << "] mV" << std::endl;
            return false;
        }
        
        double I_ion_min = I_ion_function->vector()->min();
        double I_ion_max = I_ion_function->vector()->max();
        
        if (std::isnan(I_ion_min) || std::isnan(I_ion_max) ||
            std::isinf(I_ion_min) || std::isinf(I_ion_max)) {
            std::cerr << "Error: I_ion contains NaN or Inf!" << std::endl;
            return false;
        }
        
        return true;
    }
    
    const std::vector<double>& get_cell_state(size_t node_index) const {
        if (node_index >= num_nodes) {
            throw std::out_of_range("Node index out of range");
        }
        return cell_states[node_index];
    }
    
    void print_debug_info(size_t node_index = 0) const {
        if (node_index >= num_nodes) return;
        
        std::cout << "\nNode " << node_index << " cell state summary:" << std::endl;
        std::cout << "  Na_i: " << cell_states[node_index][32] << " mM" << std::endl;
        std::cout << "  K_i: " << cell_states[node_index][33] << " mM" << std::endl;
        std::cout << "  Ca_i: " << cell_states[node_index][36] << " mM" << std::endl;
    }

private:
    std::shared_ptr<dolfin::FunctionSpace> V_space;
    size_t num_nodes;
    
    // Time step parameters
    double dt_pde_milliseconds;   // Large time step for PDE
    double dt_ode_milliseconds;   // Small time step for ODE
    int n_substeps;               // Number of ODE substeps per PDE step
    
    // Cell states
    std::vector<std::vector<double>> cell_states;
};