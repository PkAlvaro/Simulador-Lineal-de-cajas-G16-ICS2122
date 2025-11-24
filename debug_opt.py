
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator import engine
from simulator.optimizador_cajas import optimizar_cajas_grasp_saa

print("Imports successful")

day = engine.DayType.TYPE_1
print(f"Optimizing for {day}")

try:
    result = optimizar_cajas_grasp_saa(
        day_type=day,
        max_seconds=5,
        max_eval_count=2,
        context_label=f"DEBUG | Día={day.name}",
        keep_outputs_eval=False,
        use_in_memory=True,
    )
    print("Optimization successful")
    print(result)
except Exception as e:
    print(f"Optimization failed: {e}")
    import traceback
    traceback.print_exc()
