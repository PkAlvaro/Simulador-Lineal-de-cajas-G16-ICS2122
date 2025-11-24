
import sys
import os
sys.path.append(os.getcwd())

try:
    from simulator.policy_planner import plan_multi_year_optimization
    print("policy_planner import successful")
except Exception as e:
    print(f"policy_planner import failed: {e}")
