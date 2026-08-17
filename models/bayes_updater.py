"""Bayesian Updating Framework.

Sequentially updates prediction beliefs as new evidence arrives, using
Bayes' rule to combine:
  1. Prior beliefs (from regime classifier or base rates)
  2. New evidence (model predictions, feature signals, price action)
  3. Posterior = prior * likelihood / evidence

Usage in backtest:
    updater = BayesUpdater()
    posteriors = updater.update_batch(raw_probs, regime_probs, decay=0.95)
    # posteriors replace raw_probs in strategy functions
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from config.settings import BAYESIAN_UPDATING
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BayesUpdater:
    """Core Bayesian belief-updating engine.

    Maintains a running posterior belief that gets updated with each new
    piece of evidence. Supports regime-conditioned priors and time decay.
    """

    def __init__(self, config: Dict = None):
        cfg = config or BAYESIAN_UPDATING
        self.config = cfg
        self.prior_method = cfg.get("prior_method", "regime")
        self.decay_factor = cfg.get("decay_factor", 0.95)
        self.min_lr_threshold = cfg.get("likelihood_ratio_threshold", 1.5)
        self.max_updates = cfg.get("max_update_frequency", 5)

        # Regime base rates: P(UP | regime)
        self.regime_base_rates = cfg.get("regime_base_rates", {
            "bull": 0.65,
            "sideways": 0.50,
            "bear": 0.35,
        })

        # State
        self._current_prior = 0.5
        self._n_updates = 0
        self._history: List[Dict] = []

    def reset(self) -> None:
        """Reset updater state."""
        self._current_prior = 0.5
        self._n_updates = 0
        self._history.clear()

    # ── Prior Initialization ──

    def set_prior(self, prob_up: float) -> None:
        """Set the prior probability of UP movement."""
        self._current_prior = np.clip(prob_up, 0.01, 0.99)

    def set_prior_from_regime(self, regime_probs: np.ndarray) -> None:
        """Set prior from regime classifier output.

        regime_probs: [P(bear), P(sideways), P(bull)]
        """
        if len(regime_probs) != 3:
            logger.warning(f"Expected 3 regime probs, got {len(regime_probs)}")
            return
        prior = (
            regime_probs[0] * self.regime_base_rates.get("bear", 0.35)
            + regime_probs[1] * self.regime_base_rates.get("sideways", 0.50)
            + regime_probs[2] * self.regime_base_rates.get("bull", 0.65)
        )
        self._current_prior = np.clip(prior, 0.01, 0.99)

    def set_prior_from_regime_label(self, regime: str) -> None:
        """Set prior from a discrete regime label."""
        prior = self.regime_base_rates.get(regime, 0.5)
        self._current_prior = np.clip(prior, 0.01, 0.99)

    # ── Core Update ──

    def update_with_evidence(
        self,
        model_proba: float,
        evidence_strength: float = 1.0,
    ) -> float:
        """Update belief with a single piece of evidence.

        Uses Bayes' rule in log-odds space for numerical stability:
            log_odds_posterior = log_odds_prior + log_likelihood_ratio

        Args:
            model_proba: P(UP) from the model [0, 1]
            evidence_strength: Weight for this evidence [0, 1].
                1.0 = full trust, 0.0 = ignore.

        Returns:
            Posterior P(UP) [0, 1]
        """
        p = np.clip(model_proba, 0.001, 0.999)
        prior = np.clip(self._current_prior, 0.001, 0.999)

        # Convert to log-odds
        log_odds_prior = np.log(prior / (1 - prior))

        # Likelihood ratio: how much does this evidence shift belief?
        # LR = P(evidence|UP) / P(evidence|DOWN)
        # For a probability prediction, LR ≈ p / (1-p) raised to evidence_strength
        lr = (p / (1 - p)) ** evidence_strength

        # Apply minimum LR threshold — skip weak evidence
        if self.min_lr_threshold > 1.0:
            if max(lr, 1.0 / lr) < self.min_lr_threshold:
                # Evidence too weak — keep prior
                self._history.append({
                    "step": self._n_updates,
                    "prior": prior,
                    "evidence": p,
                    "lr": lr,
                    "posterior": prior,
                    "skipped": True,
                })
                self._n_updates += 1
                return prior

        log_odds_posterior = log_odds_prior + np.log(lr)
        posterior = 1.0 / (1.0 + np.exp(-log_odds_posterior))
        posterior = np.clip(posterior, 0.001, 0.999)

        self._history.append({
            "step": self._n_updates,
            "prior": prior,
            "evidence": p,
            "lr": float(lr),
            "posterior": float(posterior),
            "skipped": False,
        })
        self._n_updates += 1

        # Decay posterior toward uniform (0.5) for next step's prior
        decayed = posterior * self.decay_factor + 0.5 * (1 - self.decay_factor)
        self._current_prior = np.clip(decayed, 0.01, 0.99)

        return float(posterior)

    def update_with_regime(
        self,
        model_proba: float,
        regime_probs: np.ndarray,
        evidence_strength: float = 1.0,
    ) -> float:
        """Update: first set prior from regime, then apply evidence."""
        self.set_prior_from_regime(regime_probs)
        return self.update_with_evidence(model_proba, evidence_strength)

    # ── Batch Updates ──

    def update_batch(
        self,
        model_probas: np.ndarray,
        regime_probs: Optional[np.ndarray] = None,
        evidence_strengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized sequential update over an array of predictions.

        Each step:
          1. Optionally update prior from regime
          2. Apply Bayes' rule with model evidence
          3. Decay posterior → next step's prior

        Args:
            model_probas: Array of P(UP) predictions, shape (N,)
            regime_probs: Optional regime probs, shape (N, 3).
                If provided, prior is reset each step.
            evidence_strengths: Optional per-step weights, shape (N,).
                Defaults to confidence (max(p, 1-p)).

        Returns:
            Posterior probabilities, shape (N,)
        """
        n = len(model_probas)
        posteriors = np.zeros(n, dtype=np.float64)

        if evidence_strengths is None:
            evidence_strengths = np.maximum(model_probas, 1 - model_probas)

        for i in range(n):
            # Update prior from regime if provided
            if regime_probs is not None and i < len(regime_probs):
                self.set_prior_from_regime(regime_probs[i])

            posteriors[i] = self.update_with_evidence(
                model_proba=float(model_probas[i]),
                evidence_strength=float(evidence_strengths[i]),
            )

        return posteriors

    # ── Diagnostics ──

    def get_history(self) -> List[Dict]:
        """Return full update history."""
        return list(self._history)

    def summary(self) -> Dict:
        """Return summary statistics of updates."""
        if not self._history:
            return {"n_updates": 0}
        applied = [h for h in self._history if not h.get("skipped")]
        skipped = [h for h in self._history if h.get("skipped")]
        return {
            "n_updates": self._n_updates,
            "n_applied": len(applied),
            "n_skipped": len(skipped),
            "mean_lr": float(np.mean([h["lr"] for h in applied])) if applied else 1.0,
            "mean_prior": float(np.mean([h["prior"] for h in self._history])),
            "mean_posterior": float(np.mean([h["posterior"] for h in self._history])),
        }
