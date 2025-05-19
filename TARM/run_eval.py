import os
import subprocess
import pandas as pd
from config import Config


def RunEvaluation(base_dir, out_file, gpu_index=0):
    """
    Start a subprocess to evaluate CNN accuracy
    """
    base_dir = os.path.abspath(base_dir)
    config_inst = Config()
    
    lut_dir = os.path.join(base_dir, 'lut')
    lut_list = os.listdir(lut_dir)

    df_acc = pd.DataFrame(columns = (['mul_name', 'accuracy']))
    df_8x8 = pd.read_csv(out_file)
    if 'accuracy' in df_8x8.columns:
        print('Evaluation Stop: The output file already contains the accuracy metrics.')

    os.chdir(config_inst.application_path)
    for lut in lut_list:
        application_script = config_inst.application + '.py'
        lut_path = os.path.join(lut_dir, lut)
        cmd = ['python', application_script, config_inst.tfapprox_path, lut_path, str(gpu_index)]
        print(f"[Evaluation] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        accuracy = float(result.stdout.strip())

        mul_name = lut[:-4] # remove .bin
        new_row = [mul_name, accuracy]
        row_len = df_acc.shape[0]
        df_acc.loc[row_len] = new_row
        print(f"Multiplier: {mul_name}, Accuracy: {accuracy}")
    os.chdir(base_dir)

    merged_df = pd.merge(df_8x8, df_acc, on='mul_name')
    merged_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    eval_8x8_dir = os.path.abspath('./tmp/eval_8x8')
    out_8x8_path = os.path.abspath('./out/mul_8x8.csv')
    RunEvaluation(eval_8x8_dir, out_8x8_path, gpu_index=0)