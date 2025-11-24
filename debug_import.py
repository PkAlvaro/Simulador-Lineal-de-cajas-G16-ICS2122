
import sys
import os
sys.path.append(os.getcwd())

try:
    from simulator.cli import optimizar_cajas_grasp_saa
    print(f"optimizar_cajas_grasp_saa is: {optimizar_cajas_grasp_saa}")
except Exception as e:
    print(f"Error importing: {e}")

try:
    from simulator.optimizador_cajas import optimizar_cajas_grasp_saa as opt
    print("Direct import successful")
except Exception as e:
    print(f"Direct import failed: {e}")
