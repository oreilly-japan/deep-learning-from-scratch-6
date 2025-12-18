import matplotlib.pyplot as plt
import numpy as np

iterations = [500, 5000, 40000]
coherence_avg = [2.00, 2.60, 3.30]
coherence_std = [0.00, 0.52, 0.67]
grammar_avg = [2.10, 3.20, 3.80]
grammar_std = [0.32, 0.63, 0.92]

x = np.arange(len(iterations))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, coherence_avg, width, yerr=coherence_std,
               label='Coherence', capsize=5)
bars2 = ax.bar(x + width/2, grammar_avg, width, yerr=grammar_std,
               label='Grammar', capsize=5)

ax.set_xlabel('Iteration')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(iterations)
ax.set_ylim(0, 5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('llm_judge_scores.png')
plt.show()