# Usage example: OnePassRefineGRPOTrainer
"""
GRPO trainer with two-stage sampling during training (x->y1->y2), single-pass during inference (x->y only).

Core idea:
- Sampling: still uses (x -> y1 -> y2) to produce training samples
- Optimization: but forward and loss are computed on [x||y2] (not [x||y1||y2])
- Advantage: ΔR = R(x,y1,y2) - R(x,y1) provided by grader with group normalization
- KL/ratio: only computed on y2 segment, anchored to reference one-pass context x
"""

from trainers.onepass_refine_grpo_trainer import OnePassRefineGRPOTrainer, OnePassRefineConfig
from graders.rlvr_mlp import RLVRGrader, SimpleMLPGrader

# Example: use RLVRGrader (wrap your judge/reward model as a callable)
def my_verifier(texts: List[str]) -> List[float]:
    """
    Connect to your RLVR / RM, return scores.
    Simple example implementation.
    """
    scores = []
    for text in texts:
        # Simple heuristic scoring: check code format and length
        if "```" in text and len(text) > 50:
            scores.append(0.8)
        elif "def " in text or "class " in text:
            scores.append(0.6)
        else:
            scores.append(0.2)
    return scores

# Create grader
grader = RLVRGrader(base_callable=my_verifier)

# Configuration
config = OnePassRefineConfig(
    model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
    loss_type="dapo",
    draft_source="ref",                    # Draft source: reference model
    num_refinements_per_prompt=4,          # Number of y2 per prompt
    use_sequence_level_is=True,            # Sequence-level importance sampling
    clip_ratio_max=2.0,                    # Ratio clipping upper bound
    beta=0.01,                             # KL penalty coefficient
    
    # Optional enhancements
    use_negative_draft_push=True,          # Negative draft push
    neg_push_coef=0.25,                    # Negative push coefficient
    use_consistency_kl=True,               # Consistency KL
    consistency_kl_coef=0.1,               # Consistency KL coefficient
    
    # Base parameters
    max_completion_length=64,              # Max generation length
    do_sample=True,                        # Whether to sample
    temperature=1.0,                       # Temperature
    top_p=0.9,                            # top-p
    scale_rewards="group",                 # Reward normalization method
    reward_clip_range=(-10.0, 10.0),     # Reward clipping range
)

# Create trainer
trainer = OnePassRefineGRPOTrainer(
    grader=grader,
    model=model,                           # Current policy model
    ref_model=ref_model,                   # Reference model
    strategy=strategy,                     # Distributed strategy
    tokenizer=tokenizer,                   # Tokenizer
    optim=optimizer,                       # Optimizer
    train_dataloader=train_loader,         # Training dataloader
    eval_dataloader=eval_loader,           # Evaluation dataloader
    scheduler=scheduler,                   # Learning rate scheduler
    reward_funcs=[],                       # ΔR provided by grader
    config=config,                         # Configuration
)

# Start training
trainer.train()

# ===== Experiment suggestions =====
"""
A/B comparison experiments:

A = Original GRPOTrainer (standard single-segment GRPO)
B = OnePassRefineGRPOTrainer (two-stage sampling for training, single-pass for inference)

Ablation switches:
1. use_negative_draft_push (off/on)
2. use_consistency_kl (off/on)
3. use_sequence_level_is (token vs sequence)

Diagnostic metrics:
- mean_deltaR: average refinement advantage
- y2_active_tokens: active tokens in y2 segment
- kl_loss: KL divergence loss
- consistency_kl: consistency KL loss
- neg_push_loss: negative push loss

Offline evaluation:
- AE2 LC/WR: code pass rate
- AH: human evaluation
- MT-Bench: multi-turn dialogue evaluation
"""

# ===== Key differences summary =====
"""
Key differences from RefineGRPOTrainer:

1. Sampling strategy: Same
   - Still uses (x -> y1 -> y2) to produce training samples

2. Optimization target: Different
   - Original: optimizes π(y2|x,y1), inference requires two-stage sampling
   - This implementation: optimizes π(y|x), inference only needs single-pass

3. Forward computation: Different
   - Original: computes loss on [x||y1||y2]
   - This implementation: computes loss on [x||y2]

4. Advantage computation: Same
   - Both use ΔR = R(x,y1,y2) - R(x,y1)

5. Inference efficiency: Different
   - Original: inference requires x->y1->y2 two-stage generation
   - This implementation: inference only needs x->y single-pass generation

This distills two-stage refinement advantage into a one-pass policy without changing inference cost.
"""
