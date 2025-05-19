import os
import run_syn
import run_eval
import run_opt_2x2_tree
import build_syn_env_2x2_tree
import optimize_8x8
import build_syn_env_8x8
import final_pareto
from config import Config


config_inst = Config()
syn_2x2_dir = os.path.abspath('./tmp/syn_2x2')
syn_tree_dir = os.path.abspath('./tmp/syn_tree')
syn_8x8_dir = os.path.abspath('./tmp/syn_8x8')
out_2x2_path = os.path.abspath('./out/lut_2x2.csv')
out_tree_path = os.path.abspath('./out/lut_tree.csv')
out_8x8_path = os.path.abspath('./out/mul_8x8.csv')
eval_8x8_dir = os.path.abspath('./tmp/eval_8x8')

'''
banner = """
/-----------------------------------------------------\\
|                    TARM Framework                   |
\\-----------------------------------------------------/"""
print(banner)
print("\n=== TARM Framework Configuration ===")
print(f"• ABC executable path         : {config_inst.abc}")
print(f"• TFApprox library path       : {config_inst.tfapprox_path}")
print(f"• Application name            : {config_inst.application}")
print(f"• Application source dir      : {config_inst.application_path}")
print(f"• Distribution file           : {config_inst.dist_file}")
print(f"• Mean error constraint       : {config_inst.ME_constraint}")
print(f"• Accuracy threshold          : {config_inst.accuracy_threshold}")
print("======================================\n")
print('\n')


print("---Building Block Optimization---\n")
run_opt_2x2_tree.ParallelRun(max_workers=20)


print("---Building Block Synthesis---\n")
build_syn_env_2x2_tree.BuildSynEnv2x2()
build_syn_env_2x2_tree.BuildSynEnvTree()

run_syn.RunSynthesis(syn_2x2_dir)
run_syn.RunSynthesis(syn_tree_dir)

run_syn.MergeSynResult(syn_2x2_dir, out_2x2_path)
run_syn.MergeSynResult(syn_tree_dir, out_tree_path)


print("---Multiplier Optimization---\n")
optimize_8x8.Optimize8x8(pop_size=500, ngen=3000, cxpb=0.8, mutpb=0.2)


print("---Multiplier Synthesis---\n")
build_syn_env_8x8.BuildSynEnv8x8()

run_syn.RunSynthesis(syn_8x8_dir)
'''
run_syn.MergeSynResult(syn_8x8_dir, out_8x8_path)


print("---Multiplier Evaluation---\n")
#run_eval.RunEvaluation(eval_8x8_dir, out_8x8_path, gpu_index=0)

print("---Extract Pareto-Optimal Multipliers ---\n")
#target_hardware_cost = ['Power', 'Area'] # 1 or 2 metrics
target_hardware_cost = ['PDP'] # 1 or 2 metrics
final_pareto.FinalPareto(target_hardware_cost, './out')

