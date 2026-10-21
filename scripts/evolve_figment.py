"""Run figment's ORIGINAL evolutionary tuner (figment/evolve.py, default
settings) and save the elite MMParams for our AdaptiveMM port."""
import dataclasses, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party/figment"))
from figment.evolve import evolve

t = time.time()
best, hist = evolve(seed=0)
out = {"params": dataclasses.asdict(best),
       "history": [(h.gen, round(h.best_fitness, 2), round(h.mean_fitness, 2)) for h in hist],
       "seconds": round(time.time() - t)}
json.dump(out, open(os.path.join(ROOT, "results/figment_evolved_params.json"), "w"), indent=1)
print(json.dumps(out))
