import os
import numpy as np
import pandas as pd
from config import Config


def GetPareto(df, target_hardware_cost):
    """
    Use traversal method to find the pareto front
    """
    data = df.copy()
    data['accuracy_inv'] = -data['accuracy']
    cols = ['accuracy_inv'] + target_hardware_cost
    vals = data[cols].values
    n_points = vals.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if is_pareto[i]:
            c = vals[i]
            is_dominated = np.all(c <= vals, axis=1) & np.any(c < vals, axis=1)
            is_pareto[is_dominated] = False
    return df[is_pareto]


def FinalPareto(target_hardware_cost, out_path):
    """
    Construct the final Pareto front based on metrics obtained from simulation and synthesis
    """
    config_inst = Config()
    df_8x8 = pd.read_csv('./out/mul_8x8.csv')
    df_8x8.rename(columns={'area' : 'Area', 'power' : 'Power', 'delay' : 'Delay'}, inplace=True)
    
    if 'PDP' not in df_8x8.columns:
        df_8x8['PDP'] = df_8x8['Power'] * df_8x8['Delay']

    if 'ADP' not in df_8x8.columns:
        df_8x8['ADP'] = df_8x8['Area'] * df_8x8['Delay']

    df_8x8 = df_8x8.loc[df_8x8.loc[:, 'accuracy'] >= config_inst.accuracy_threshold, :].copy() 
    df_pareto = GetPareto(df_8x8, target_hardware_cost)
    out_file = os.path.join(out_path, 'mul_8x8_pareto.csv')
    df_pareto.to_csv(out_file, index=False)
    print(f"[FINISH] Final Pareto-optimal solutions have been written to {out_file}")


if __name__ == '__main__':
    #target_hardware_cost = ['Area'] # 1 or 2 metrics
    target_hardware_cost = ['Power', 'Area'] # 1 or 2 metrics

    FinalPareto(target_hardware_cost, './out')