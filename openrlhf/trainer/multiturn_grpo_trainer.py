"""
Multi-Turn GRPO Trainer

Multi-turn generation↔verification trainer based on existing GRPO trainer.
Implements Gen₁→Ver₁→Gen₂→Ver₂→... multi-turn training mechanism.
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel
import numpy as np
import copy

from .grpo_trainer import GRPOTrainer, GRPOConfig


@dataclass
class MultiTurnGRPOConfig(GRPOConfig):
    """Multi-turn GRPO configuration"""
    # Multi-turn parameters
    max_train_turns: int = 3  # Max training turns (Gen/Ver each counts as one turn)
    max_infer_turns: int = 6  # Inference max depth
    use_turn_aware_advantage: bool = True  # Turn-Aware PPO
    gen_abs_weight: float = 0.0  # r_abs weight (default 0 in paper)
    gen_imp_weight: float = 1.0  # r_imp weight (default 1 in paper)
    ver_weight: float = 1.0  # Verification turn weight
    stop_when_pass_1: bool = True  # Early stop when pass rate = 1
    
    # Monte Carlo merging (copy scalar turn return to all tokens in turn)
    gamma: float = 1.0
    lam: float = 1.0
    
    # Behavior policy & reference policy
    use_old_model_snapshot: bool = True  # PPO denominator: explicit old policy snapshot
    old_model_sync_frequency: int = 4  # Sync old_model every N steps


class MultiTurnGRPOTrainer(GRPOTrainer):
    """
    Multi-turn GRPO trainer.
    
    Extends existing GRPO trainer to support multi-turn generation↔verification training.
    """
    
    def __init__(self, model, ref_model, tokenizer, config: MultiTurnGRPOConfig, strategy=None, **kwargs):
        """
        Initialize multi-turn GRPO trainer.
        
        Args:
            model: Main model
            ref_model: Reference model
            tokenizer: Tokenizer
            config: Multi-turn GRPO configuration
            strategy: Training strategy
        """
        # Initialize parent class with multi-turn config
        super().__init__(model, ref_model, tokenizer, config, strategy, **kwargs)
        
        # Multi-turn specific initialization
        self.multiturn_config = config
        self.turn_tags = {
            "gen_open": "<GEN>\n",
            "gen_close": "</GEN>\n",
            "ver_open": "<VER>\n",
            "ver_close": "</VER>\n",
            "tool_feedback_open": "<TOOL>\n",
            "tool_feedback_close": "</TOOL>\n",
        }
        
        # Turn counter
        self.turn_count = 0
        
    def rollout_turns(self, prompts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Multi-turn generation↔verification rollout.
        
        Returns:
            List[List[Dict]]: Turn sequence for each prompt
            turns[p] = [
                {"type":"gen","segment":"…y_gen1…","pass_before":0.0,"pass_after":p1},
                {"type":"ver","segment":"…y_ver1…","pass_before":p1,"pass_after":p1'},
                {"type":"gen","segment":"…y_gen2…","pass_before":p1',"pass_after":p2},
                ...
            ]
        """
        all_prompts_turns = []
        
        for x in prompts:
            ctx = f"User:\n{x}\n\n"
            pass_hist = []  # Store passrate after Gen turns only (for improvement)
            turns = []
            
            for k in range(self.multiturn_config.max_train_turns):
                # ---- Gen_k ----
                gen_in = ctx + self.turn_tags["gen_open"]
                yk = self._sample(
                    self.model if k > 0 else self.ref_model, 
                    gen_in,
                    max_new_tokens=self.config.max_completion_length
                )
                ctx += self.turn_tags["gen_open"] + yk + self.turn_tags["gen_close"]
                
                # Executor: generate/reuse tests + run, get passrate_k
                pr_k, tool_fb = self._evaluate_and_feedback(x, ctx, yk, turn=f"gen_{k+1}")
                turns.append(dict(
                    type="gen", 
                    segment=yk,
                    pass_before=pass_hist[-1] if pass_hist else 0.0,
                    pass_after=pr_k,
                    tool_feedback=tool_fb
                ))
                pass_hist.append(pr_k)
                
                if self.multiturn_config.stop_when_pass_1 and pr_k >= 1.0:
                    break
                
                # ---- Ver_k ----
                ver_in = ctx + self.turn_tags["ver_open"]
                vk = self._sample(
                    self.model, 
                    ver_in,
                    max_new_tokens=self.config.max_completion_length
                )
                ctx += self.turn_tags["ver_open"] + vk + self.turn_tags["ver_close"]
                
                # Verification turn: produce tests (or explanation) + execute to get pass rate
                prk_ver, tool_fb2 = self._evaluate_and_feedback(
                    x, ctx, vk, turn=f"ver_{k+1}", is_verification=True
                )
                turns.append(dict(
                    type="ver", 
                    segment=vk,
                    pass_before=None,
                    pass_after=prk_ver,
                    tool_feedback=tool_fb2
                ))
            
            all_prompts_turns.append(turns)
        
        return all_prompts_turns
    
    def compute_turn_rewards(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute turn-level rewards.
        
        Args:
            turns: Turn list for a single prompt
            
        Returns:
            Turn list with updated reward_scalar
        """
        abs_w, imp_w = self.multiturn_config.gen_abs_weight, self.multiturn_config.gen_imp_weight
        gen_pass_hist = []
        
        for i, t in enumerate(turns):
            if t["type"] == "gen":
                pr = float(t["pass_after"])
                pr_prev2 = gen_pass_hist[-1] if len(gen_pass_hist) >= 1 else 0.0
                r = abs_w * pr + imp_w * (pr - pr_prev2)
                t["reward_scalar"] = r
                gen_pass_hist.append(pr)
            else:
                r = float(t["pass_after"])
                t["reward_scalar"] = self.multiturn_config.ver_weight * r
        
        # Turn-aware backfill: add Gen_k reward to Ver_{k-1}
        if self.multiturn_config.use_turn_aware_advantage:
            for i, t in enumerate(turns):
                if t["type"] == "gen":
                    j = i - 1
                    if j >= 0 and turns[j]["type"] == "ver":
                        turns[j]["reward_scalar"] += t["reward_scalar"]
        
        return turns
    
    def prepare_turn_batch(self, prompts: List[str], all_turns: List[List[Dict[str, Any]]]) -> Tuple:
        """
        Build training samples for all turns of all prompts.
        
        Returns:
            input_ids, attention_mask, loss_mask, labels, advantages
        """
        ids_list, attn_list, mask_list, labels_list, adv_list = [], [], [], [], []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        MAX = 512 + self.config.max_completion_length
        
        for x, turns in zip(prompts, all_turns):
            ctx_left = f"User:\n{x}\n\n"
            
            for t in turns:
                # 1) Build context + current turn segment
                if t["type"] == "gen":
                    seg_open, seg_close = self.turn_tags["gen_open"], self.turn_tags["gen_close"]
                else:
                    seg_open, seg_close = self.turn_tags["ver_open"], self.turn_tags["ver_close"]
                
                # Input = (ctx_left + seg_open) -> generate seg -> seg_close only in labels
                inp_text = ctx_left + seg_open
                seg_text = t["segment"]
                full_text = inp_text + seg_text + seg_close
                
                ids_full = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
                ids_inp = self.tokenizer(inp_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
                
                start = ids_inp.numel()
                end = start + self.tokenizer(seg_text, add_special_tokens=False, return_tensors="pt").input_ids[0].numel()
                
                # Left-truncate to fit MAX
                if ids_full.numel() > MAX:
                    cut = ids_full.numel() - MAX
                    ids_full = ids_full[cut:]
                    start = max(0, start - cut)
                    end = max(start, end - cut)
                
                attn = torch.ones_like(ids_full)
                loss_mask = torch.zeros_like(ids_full)
                loss_mask[start:end] = 1
                
                ids_list.append(ids_full)
                attn_list.append(attn)
                labels_list.append(ids_full.clone())
                mask_list.append(loss_mask)
                adv_list.append(float(t["reward_scalar"]))
                
                # 2) Update ctx_left: append current turn segment with TOOL feedback
                ctx_left = full_text
                if t.get("tool_feedback"):
                    ctx_left += (self.turn_tags["tool_feedback_open"] + 
                               t["tool_feedback"] + 
                               self.turn_tags["tool_feedback_close"])
        
        # pad & stack & device
        max_len = max(x.numel() for x in ids_list)
        B = len(ids_list)
        
        def pad_stack(lst, fill):
            out = torch.full((B, max_len), fill, dtype=torch.long)
            for i, x in enumerate(lst):
                out[i, :x.numel()] = x
            return out
        
        input_ids = pad_stack(ids_list, pad_id)
        attention_mask = pad_stack(attn_list, 0)
        loss_mask = pad_stack(mask_list, 0)
        labels = pad_stack(labels_list, pad_id)
        advantages = torch.tensor(adv_list, dtype=torch.float32)
        
        device = self.model.device
        return (input_ids.to(device), attention_mask.to(device),
                loss_mask.to(device), labels.to(device), advantages.to(device))
    
    def train_step(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Multi-turn training step.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            Training metrics dict
        """
        # 1) Rollout multi-turn
        all_turns = self.rollout_turns(prompts)
        
        # 2) Turn-level rewards
        all_turns = [self.compute_turn_rewards(t) for t in all_turns]
        
        # 3) Build turn segment samples
        batch = self.prepare_turn_batch(prompts, all_turns)
        
        # 4) Compute loss (using parent's compute_refinement_loss)
        loss_dict = self.compute_refinement_loss(*batch)
        loss_dict["total_loss"].backward()
        
        # 5) Sync old_model
        if self.step_count % self.multiturn_config.old_model_sync_frequency == 0:
            self._sync_old_model()
        
        # 6) Statistics
        mean_adv = batch[-1].mean().item()
        return dict(
            total_loss=loss_dict["total_loss"].item(),
            policy_loss=loss_dict["policy_loss"].item(),
            kl_loss=loss_dict["kl_loss"].item(),
            mean_ratio=loss_dict["mean_ratio"].item(),
            mean_advantage=mean_adv,
            active_tokens=int((batch[2][:,1:].float()).sum().item()),
            turn_count=self.turn_count,
        )
    
    def inference(self, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
        """
        Multi-turn inference (test-time scaling).
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Max new tokens
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for x in prompts:
            ctx = f"User:\n{x}\n\n"
            best_code, best_pass = "", 0.0
            
            for k in range(self.multiturn_config.max_infer_turns):
                # Gen_k
                gen_in = ctx + self.turn_tags["gen_open"]
                yk = self._sample(self.model, gen_in, max_new_tokens=max_new_tokens//2)
                ctx += self.turn_tags["gen_open"] + yk + self.turn_tags["gen_close"]
                
                pr_k, tool_fb = self._evaluate_and_feedback(x, ctx, yk, turn=f"gen_{k+1}")
                if pr_k > best_pass:
                    best_pass, best_code = pr_k, yk
                if self.multiturn_config.stop_when_pass_1 and pr_k >= 1.0:
                    break
                
                # Ver_k
                ver_in = ctx + self.turn_tags["ver_open"]
                vk = self._sample(self.model, ver_in, max_new_tokens=max_new_tokens//2)
                ctx += self.turn_tags["ver_open"] + vk + self.turn_tags["ver_close"]
                
                prk_ver, tool_fb2 = self._evaluate_and_feedback(
                    x, ctx, vk, turn=f"ver_{k+1}", is_verification=True
                )
                
                # Tool feedback can be appended to ctx as next turn prompt
                ctx += (self.turn_tags["tool_feedback_open"] + tool_fb + tool_fb2 +
                       self.turn_tags["tool_feedback_close"])
            
            responses.append(best_code if best_code else yk)
        
        return responses