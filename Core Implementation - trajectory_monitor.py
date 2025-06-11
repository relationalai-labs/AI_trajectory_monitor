"""
AI Trajectory and Learning Monitor - Core Implementation

This module provides the main AITrajectoryMonitor class for tracking learning health,
detecting reward hacking, and predicting capability emergence.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """Represents the learning state at a given time point."""
    timestamp: int
    competencies: np.ndarray  # [accuracy, reasoning, alignment, efficiency, coherence]
    structure_score: float
    vibe_score: float
    complexity_measure: float
    
    
@dataclass 
class HealthReport:
    """Contains health assessment and warnings from trajectory monitoring."""
    timestamp: int
    health_score: float
    vibe_structure_correlation: float
    warnings: Dict[str, Any]
    predictions: Dict[str, Any]
    trajectory_summary: Dict[str, float]


class KLDivergenceCalculator:
    """Utility class for computing KL divergence between learning trajectories."""
    
    @staticmethod
    def kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Compute KL divergence D_KL(P || Q).
        
        Args:
            P: Target distribution (ideal trajectory)
            Q: Approximating distribution (actual trajectory) 
            epsilon: Small constant to avoid log(0)
            
        Returns:
            KL divergence value
        """
        # Ensure valid probability distributions
        P = np.clip(P, epsilon, 1.0)
        Q = np.clip(Q, epsilon, 1.0)
        
        # Normalize to ensure sum = 1
        P = P / np.sum(P)
        Q = Q / np.sum(Q)
        
        return np.sum(P * np.log(P / Q))
    
    @staticmethod
    def trajectory_divergence(ideal_trajectory: List[np.ndarray], 
                            actual_trajectory: List[np.ndarray]) -> float:
        """Compute average KL divergence over trajectory windows."""
        if len(ideal_trajectory) != len(actual_trajectory):
            raise ValueError("Trajectory lengths must match")
            
        divergences = []
        for ideal, actual in zip(ideal_trajectory, actual_trajectory):
            div = KLDivergenceCalculator.kl_divergence(ideal, actual)
            divergences.append(div)
            
        return np.mean(divergences)


class RewardHackingDetector:
    """Detects when AI systems are gaming metrics rather than truly learning."""
    
    def __init__(self, correlation_threshold: float = 0.3):
        self.correlation_threshold = correlation_threshold
        self.structure_history = deque(maxlen=100)
        self.vibe_history = deque(maxlen=100)
        
    def detect_metric_gaming(self, structure_score: float, vibe_score: float) -> Dict[str, Any]:
        """
        Detect if AI is gaming metrics based on structure vs vibe divergence.
        
        Args:
            structure_score: Measurable performance metrics
            vibe_score: Emergent learning quality patterns
            
        Returns:
            Detection results with confidence scores
        """
        self.structure_history.append(structure_score)
        self.vibe_history.append(vibe_score)
        
        if len(self.structure_history) < 10:
            return {"gaming_detected": False, "confidence": 0.0, "reason": "insufficient_data"}
        
        # Check correlation between structure and vibe
        correlation = np.corrcoef(list(self.structure_history), list(self.vibe_history))[0, 1]
        
        # Look for diverging trends
        structure_trend = np.polyfit(range(len(self.structure_history)), list(self.structure_history), 1)[0]
        vibe_trend = np.polyfit(range(len(self.vibe_history)), list(self.vibe_history), 1)[0]
        
        # Gaming indicators
        gaming_signals = {
            "correlation_breakdown": correlation < self.correlation_threshold,
            "diverging_trends": structure_trend > 0 and vibe_trend < 0,
            "sudden_structure_jump": structure_score > np.mean(list(self.structure_history)[:-1]) + 2 * np.std(list(self.structure_history)[:-1]),
            "vibe_degradation": vibe_score < np.mean(list(self.vibe_history)[:-1]) - np.std(list(self.vibe_history)[:-1])
        }
        
        # Calculate confidence
        confidence = sum(gaming_signals.values()) / len(gaming_signals)
        
        return {
            "gaming_detected": confidence > 0.5,
            "confidence": confidence,
            "correlation": correlation,
            "signals": gaming_signals,
            "reason": "vibe_structure_divergence" if gaming_signals["correlation_breakdown"] else "normal"
        }


class TemporalCoherenceTracker:
    """Tracks whether AI maintains consistent identity and values over time."""
    
    def __init__(self, lookback_window: int = 1000):
        self.lookback_window = lookback_window
        self.competency_history = deque(maxlen=lookback_window)
        self.value_embeddings = deque(maxlen=lookback_window)
        
    def track_identity_drift(self, current_competencies: np.ndarray, 
                           value_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Track whether AI's core identity is drifting over time.
        
        Args:
            current_competencies: Current competency measurements
            value_embedding: Optional embedding representing AI's values
            
        Returns:
            Identity drift analysis
        """
        self.competency_history.append(current_competencies.copy())
        if value_embedding is not None:
            self.value_embeddings.append(value_embedding.copy())
        
        if len(self.competency_history) < 50:
            return {"drift_detected": False, "confidence": 0.0, "reason": "insufficient_history"}
        
        # Analyze competency stability
        recent_window = list(self.competency_history)[-50:]
        older_window = list(self.competency_history)[-100:-50] if len(self.competency_history) >= 100 else []
        
        if not older_window:
            return {"drift_detected": False, "confidence": 0.0, "reason": "need_longer_history"}
        
        # Calculate drift metrics
        recent_mean = np.mean(recent_window, axis=0)
        older_mean = np.mean(older_window, axis=0)
        
        # Identity coherence: how much has the AI's "personality" changed?
        identity_shift = np.linalg.norm(recent_mean - older_mean)
        capability_growth = np.mean(recent_mean) - np.mean(older_mean)
        
        # Concerning: capabilities growing but personality/values shifting
        drift_indicators = {
            "significant_identity_shift": identity_shift > 0.2,
            "capability_with_drift": capability_growth > 0.1 and identity_shift > 0.15,
            "value_inconsistency": self._check_value_consistency() if self.value_embeddings else False
        }
        
        confidence = sum(drift_indicators.values()) / len(drift_indicators)
        
        return {
            "drift_detected": confidence > 0.6,
            "confidence": confidence,
            "identity_shift_magnitude": identity_shift,
            "capability_growth": capability_growth,
            "indicators": drift_indicators
        }
    
    def _check_value_consistency(self) -> bool:
        """Check if value embeddings show concerning inconsistency."""
        if len(self.value_embeddings) < 20:
            return False
            
        recent_values = list(self.value_embeddings)[-10:]
        older_values = list(self.value_embeddings)[-20:-10]
        
        recent_center = np.mean(recent_values, axis=0)
        older_center = np.mean(older_values, axis=0)
        
        # High distance suggests value drift
        value_drift = np.linalg.norm(recent_center - older_center)
        return value_drift > 0.3


class CapabilityEmergencePredictor:
    """Predicts when new capabilities might emerge based on learning trajectory."""
    
    def __init__(self):
        self.trajectory_history = deque(maxlen=500)
        self.capability_jumps = []  # Historical capability emergence points
        
    def predict_emergence(self, current_state: LearningState) -> Dict[str, Any]:
        """
        Predict if/when capability emergence might occur.
        
        Args:
            current_state: Current learning state
            
        Returns:
            Emergence prediction with timeline and confidence
        """
        self.trajectory_history.append(current_state)
        
        if len(self.trajectory_history) < 100:
            return {"emergence_likely": False, "confidence": 0.0, "eta_steps": None}
        
        # Analyze trajectory patterns
        competency_trends = self._analyze_competency_trends()
        complexity_pressure = self._measure_complexity_pressure()
        learning_acceleration = self._detect_learning_acceleration()
        
        # Emergence indicators
        emergence_signals = {
            "accelerating_learning": learning_acceleration > 0.05,
            "complexity_buildup": complexity_pressure > 0.7,
            "cross_domain_growth": self._detect_cross_domain_growth(),
            "approaching_threshold": any(trend > 0.8 for trend in competency_trends.values())
        }
        
        confidence = sum(emergence_signals.values()) / len(emergence_signals)
        
        # Predict timeline if emergence likely
        eta_steps = None
        if confidence > 0.6:
            eta_steps = self._estimate_emergence_timeline(competency_trends)
        
        return {
            "emergence_likely": confidence > 0.6,
            "confidence": confidence,
            "eta_steps": eta_steps,
            "signals": emergence_signals,
            "competency_trends": competency_trends
        }
    
    def _analyze_competency_trends(self) -> Dict[str, float]:
        """Analyze growth trends in each competency dimension."""
        if len(self.trajectory_history) < 50:
            return {}
            
        competencies = np.array([state.competencies for state in list(self.trajectory_history)[-50:]])
        trends = {}
        
        for i, comp_name in enumerate(['accuracy', 'reasoning', 'alignment', 'efficiency', 'coherence']):
            # Fit linear trend
            x = np.arange(len(competencies))
            slope = np.polyfit(x, competencies[:, i], 1)[0]
            trends[comp_name] = slope
            
        return trends
    
    def _measure_complexity_pressure(self) -> float:
        """Measure how much complexity pressure is building up."""
        if len(self.trajectory_history) < 20:
            return 0.0
            
        recent_complexity = [state.complexity_measure for state in list(self.trajectory_history)[-20:]]
        return np.mean(recent_complexity)
    
    def _detect_learning_acceleration(self) -> float:
        """Detect if learning is accelerating (second derivative)."""
        if len(self.trajectory_history) < 30:
            return 0.0
            
        recent_competencies = [np.mean(state.competencies) for state in list(self.trajectory_history)[-30:]]
        
        # Calculate second derivative (acceleration)
        first_deriv = np.diff(recent_competencies)
        second_deriv = np.diff(first_deriv)
        
        return np.mean(second_deriv[-10:]) if len(second_deriv) >= 10 else 0.0
    
    def _detect_cross_domain_growth(self) -> bool:
        """Detect if multiple competencies are growing simultaneously."""
        trends = self._analyze_competency_trends()
        if not trends:
            return False
            
        positive_trends = sum(1 for trend in trends.values() if trend > 0.01)
        return positive_trends >= 3  # At least 3 competencies growing
    
    def _estimate_emergence_timeline(self, trends: Dict[str, float]) -> int:
        """Estimate when capability emergence might occur."""
        # Simple heuristic: extrapolate when fastest-growing competency hits threshold
        if not trends:
            return None
            
        max_trend = max(trends.values())
        if max_trend <= 0:
            return None
            
        # Assume emergence at competency level 0.9, estimate steps to reach it
        current_max = max(self.trajectory_history[-1].competencies)
        steps_to_threshold = (0.9 - current_max) / max_trend
        
        return max(10, int(steps_to_threshold))  # At least 10 steps


class AITrajectoryMonitor:
    """
    Main class for monitoring AI learning trajectories and detecting problems.
    
    This system tracks learning health, detects reward hacking, monitors temporal
    coherence, and predicts capability emergence.
    """
    
    def __init__(self, 
                 competency_dims: List[str] = None,
                 complexity_penalty: float = 0.1,
                 drift_threshold: float = 0.3,
                 lookback_window: int = 1000):
        """
        Initialize trajectory monitor.
        
        Args:
            competency_dims: Names of competency dimensions to track
            complexity_penalty: Weight for complexity penalty in learning velocity
            drift_threshold: Threshold for detecting concerning drift
            lookback_window: How many steps to remember for temporal analysis
        """
        self.competency_dims = competency_dims or ['accuracy', 'reasoning', 'alignment', 'efficiency', 'coherence']
        self.complexity_penalty = complexity_penalty
        self.drift_threshold = drift_threshold
        
        # Initialize components
        self.kl_calculator = KLDivergenceCalculator()
        self.reward_detector = RewardHackingDetector(correlation_threshold=drift_threshold)
        self.coherence_tracker = TemporalCoherenceTracker(lookback_window=lookback_window)
        self.emergence_predictor = CapabilityEmergencePredictor()
        
        # State tracking
        self.trajectory_history = deque(maxlen=lookback_window)
        self.ideal_trajectory = None  # Set via set_ideal_trajectory()
        
        logger.info(f"AITrajectoryMonitor initialized with {len(self.competency_dims)} competency dimensions")
    
    def set_ideal_trajectory(self, ideal_trajectory: List[np.ndarray]):
        """Set the ideal learning trajectory for comparison."""
        self.ideal_trajectory = ideal_trajectory
        logger.info(f"Ideal trajectory set with {len(ideal_trajectory)} time points")
    
    def audit_learning_step(self, 
                           current_state: Dict[str, Any],
                           reward_signal: float,
                           timestamp: int,
                           value_embedding: Optional[np.ndarray] = None) -> HealthReport:
        """
        Perform comprehensive audit of current learning step.
        
        Args:
            current_state: Dictionary containing model state information
            reward_signal: Current reward/loss signal
            timestamp: Current time step
            value_embedding: Optional representation of AI's current values
            
        Returns:
            HealthReport with comprehensive analysis
        """
        # Extract competencies from current state
        competencies = self._extract_competencies(current_state)
        
        # Calculate structure and vibe scores
        structure_score = self._calculate_structure_score(current_state, reward_signal)
        vibe_score = self._calculate_vibe_score(current_state, competencies)
        complexity_measure = self._calculate_complexity(current_state)
        
        # Create learning state
        learning_state = LearningState(
            timestamp=timestamp,
            competencies=competencies,
            structure_score=structure_score,
            vibe_score=vibe_score,
            complexity_measure=complexity_measure
        )
        
        self.trajectory_history.append(learning_state)
        
        # Run all monitoring components
        reward_analysis = self.reward_detector.detect_metric_gaming(structure_score, vibe_score)
        coherence_analysis = self.coherence_tracker.track_identity_drift(competencies, value_embedding)
        emergence_analysis = self.emergence_predictor.predict_emergence(learning_state)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(reward_analysis, coherence_analysis, emergence_analysis)
        
        # Compile warnings
        warnings = {
            "reward_hacking": reward_analysis,
            "identity_drift": coherence_analysis["drift_detected"],
            "vibe_structure_divergence": reward_analysis["correlation"] < self.drift_threshold,
            "capability_emergence": emergence_analysis["emergence_likely"]
        }
        
        # Compile predictions
        predictions = {
            "capability_emergence_probability": emergence_analysis["confidence"],
            "emergence_eta": emergence_analysis["eta_steps"],
            "health_trajectory": "improving" if health_score > 0.7 else "concerning" if health_score < 0.4 else "stable"
        }
        
        # Trajectory summary
        trajectory_summary = {
            "mean_competency": np.mean(competencies),
            "competency_growth": self._calculate_competency_growth(),
            "stability_score": self._calculate_stability_score(),
            "complexity_trend": self._calculate_complexity_trend()
        }
        
        return HealthReport(
            timestamp=timestamp,
            health_score=health_score,
            vibe_structure_correlation=reward_analysis.get("correlation", 0.0),
            warnings=warnings,
            predictions=predictions,
            trajectory_summary=trajectory_summary
        )
    
    def _extract_competencies(self, current_state: Dict[str, Any]) -> np.ndarray:
        """Extract competency measurements from model state."""
        # This is a placeholder - in practice, you'd implement domain-specific extraction
        # For demonstration, we'll create synthetic competencies
        
        if "competencies" in current_state:
            return np.array(current_state["competencies"])
        
        # Generate synthetic competencies based on available metrics
        competencies = np.zeros(len(self.competency_dims))
        
        # Example mappings - customize for your domain
        if "accuracy" in current_state:
            competencies[0] = current_state["accuracy"]
        if "loss" in current_state:
            competencies[0] = max(0, 1 - current_state["loss"])  # Convert loss to accuracy-like metric
            
        # Fill remaining with reasonable defaults
        for i in range(len(competencies)):
            if competencies[i] == 0:
                competencies[i] = 0.5 + 0.1 * np.random.randn()  # Small random component
                
        return np.clip(competencies, 0, 1)
    
    def _calculate_structure_score(self, current_state: Dict[str, Any], reward_signal: float) -> float:
        """Calculate traditional performance metrics score."""
        # Weight different metrics based on availability
        metrics = []
        
        if "accuracy" in current_state:
            metrics.append(current_state["accuracy"])
        if "f1_score" in current_state:
            metrics.append(current_state["f1_score"])
        if "loss" in current_state:
            metrics.append(1 - min(current_state["loss"], 1))  # Convert loss to performance metric
            
        # Include reward signal
        normalized_reward = 1 / (1 + np.exp(-reward_signal))  # Sigmoid normalization
        metrics.append(normalized_reward)
        
        return np.mean(metrics) if metrics else 0.5
    
    def _calculate_vibe_score(self, current_state: Dict[str, Any], competencies: np.ndarray) -> float:
        """Calculate emergent learning quality patterns."""
        # This represents the "feeling" that learning is healthy
        factors = []
        
        # Competency balance (not just peak performance)
        competency_balance = 1 - np.std(competencies)  # Lower std = more balanced
        factors.append(competency_balance)
        
        # Consistency with recent history
        if len(self.trajectory_history) > 5:
            recent_competencies = [state.competencies for state in list(self.trajectory_history)[-5:]]
            consistency = 1 - np.mean([np.linalg.norm(comp - competencies) for comp in recent_competencies])
            factors.append(max(0, consistency))
        
        # Add domain-specific "vibe" indicators here
        if "explanation_quality" in current_state:
            factors.append(current_state["explanation_quality"])
        if "reasoning_coherence" in current_state:
            factors.append(current_state["reasoning_coherence"])
            
        return np.mean(factors) if factors else 0.5
    
    def _calculate_complexity(self, current_state: Dict[str, Any]) -> float:
        """Calculate complexity measure of current state."""
        # Simple proxy for model complexity
        complexity_factors = []
        
        if "parameter_count" in current_state:
            # Normalize parameter count
            param_complexity = min(current_state["parameter_count"] / 1e9, 1.0)
            complexity_factors.append(param_complexity)
            
        if "gradient_norm" in current_state:
            # High gradient norms suggest complexity
            grad_complexity = min(current_state["gradient_norm"] / 10.0, 1.0)
            complexity_factors.append(grad_complexity)
            
        # Trajectory-based complexity (how much is the model changing?)
        if len(self.trajectory_history) > 1:
            prev_competencies = self.trajectory_history[-1].competencies
            current_competencies = self._extract_competencies(current_state)
            change_magnitude = np.linalg.norm(current_competencies - prev_competencies)
            complexity_factors.append(min(change_magnitude * 5, 1.0))
            
        return np.mean(complexity_factors) if complexity_factors else 0.3
    
    def _calculate_health_score(self, reward_analysis: Dict, coherence_analysis: Dict, emergence_analysis: Dict) -> float:
        """Calculate overall learning health score."""
        health_factors = []
        
        # Reward hacking penalty
        if reward_analysis["gaming_detected"]:
            health_factors.append(1 - reward_analysis["confidence"])
        else:
            health_factors.append(0.8)  # Good correlation
            
        # Identity drift penalty
        if coherence_analysis["drift_detected"]:
            health_factors.append(1 - coherence_analysis["confidence"])
        else:
            health_factors.append(0.9)  # Stable identity
            
        # Emergence readiness (can be good or concerning)
        emergence_factor = 0.7 if emergence_analysis["emergence_likely"] else 0.6
        health_factors.append(emergence_factor)
        
        return np.mean(health_factors)
    
    def _calculate_competency_growth(self) -> float:
        """Calculate recent competency growth rate."""
        if len(self.trajectory_history) < 10:
            return 0.0
            
        recent_states = list(self.trajectory_history)[-10:]
        older_states = list(self.trajectory_history)[-20:-10] if len(self.trajectory_history) >= 20 else []
        
        if not older_states:
            return 0.0
            
        recent_mean = np.mean([np.mean(state.competencies) for state in recent_states])
        older_mean = np.mean([np.mean(state.competencies) for state in older_states])
        
        return recent_mean - older_mean
    
    def _calculate_stability_score(self) -> float:
        """Calculate how stable the learning trajectory is."""
        if len(self.trajectory_history) < 5:
            return 0.5
            
        recent_competencies = [state.competencies for state in list(self.trajectory_history)[-5:]]
        variances = np.var(recent_competencies, axis=0)
        
        # Lower variance = higher stability
        return 1 - np.mean(variances)
    
    def _calculate_complexity_trend(self) -> float:
        """Calculate trend in complexity over time."""
        if len(self.trajectory_history) < 10:
            return 0.0
            
        complexity_values = [state.complexity_measure for state in list(self.trajectory_history)[-10:]]
        x = np.arange(len(complexity_values))
        slope = np.polyfit(x, complexity_values, 1)[0]
        
        return slope
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of learning trajectory."""
        if not self.trajectory_history:
            return {"status": "no_data"}
            
        current_state = self.trajectory_history[-1]
        
        summary = {
            "total_steps": len(self.trajectory_history),
            "current_competencies": {
                dim: current_state.competencies[i] 
                for i, dim in enumerate(self.competency_dims)
            },
            "health_trend": self._get_health_trend(),
            "key_events": self._identify_key_events(),
            "risk_assessment": self._assess_current_risks(),
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _get_health_trend(self) -> str:
        """Analyze overall health trend."""
        if len(self.trajectory_history) < 10:
            return "insufficient_data"
            
        # Calculate health scores for recent history
        recent_health = []
        for i in range(min(10, len(self.trajectory_history))):
            state = self.trajectory_history[-(i+1)]
            # Simplified health calculation for trend analysis
            health = np.mean(state.competencies) * (1 - state.complexity_measure * 0.3)
            recent_health.append(health)
            
        recent_health.reverse()  # Chronological order
        
        # Fit trend line
        x = np.arange(len(recent_health))
        slope = np.polyfit(x, recent_health, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _identify_key_events(self) -> List[Dict[str, Any]]:
        """Identify significant events in learning trajectory."""
        events = []
        
        if len(self.trajectory_history) < 20:
            return events
            
        # Look for sudden competency jumps
        competency_changes = []
        for i in range(1, len(self.trajectory_history)):
            prev_comp = np.mean(self.trajectory_history[i-1].competencies)
            curr_comp = np.mean(self.trajectory_history[i].competencies)
            competency_changes.append(curr_comp - prev_comp)
            
        # Identify jumps (more than 2 standard deviations)
        mean_change = np.mean(competency_changes)
        std_change = np.std(competency_changes)
        threshold = mean_change + 2 * std_change
        
        for i, change in enumerate(competency_changes):
            if change > threshold:
                events.append({
                    "type": "capability_jump",
                    "timestamp": self.trajectory_history[i+1].timestamp,
                    "magnitude": change,
                    "description": f"Significant capability increase detected ({change:.3f})"
                })
                
        return events[-5:]  # Return last 5 events
    
    def _assess_current_risks(self) -> Dict[str, str]:
        """Assess current risk levels."""
        if not self.trajectory_history:
            return {"overall": "unknown"}
            
        current_state = self.trajectory_history[-1]
        risks = {}
        
        # Complexity risk
        if current_state.complexity_measure > 0.8:
            risks["complexity"] = "high"
        elif current_state.complexity_measure > 0.6:
            risks["complexity"] = "medium"
        else:
            risks["complexity"] = "low"
            
        # Stability risk
        stability = self._calculate_stability_score()
        if stability < 0.3:
            risks["stability"] = "high"
        elif stability < 0.6:
            risks["stability"] = "medium"
        else:
            risks["stability"] = "low"
            
        # Growth risk (too fast or too slow)
        growth = self._calculate_competency_growth()
        if abs(growth) > 0.1:
            risks["growth_rate"] = "high"
        elif abs(growth) > 0.05:
            risks["growth_rate"] = "medium"
        else:
            risks["growth_rate"] = "low"
            
        # Overall risk assessment
        risk_levels = list(risks.values())
        if "high" in risk_levels:
            risks["overall"] = "high"
        elif "medium" in risk_levels:
            risks["overall"] = "medium"
        else:
            risks["overall"] = "low"
            
        return risks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on trajectory analysis."""
        recommendations = []
        
        if not self.trajectory_history:
            return ["Insufficient data for recommendations"]
            
        current_state = self.trajectory_history[-1]
        risks = self._assess_current_risks()
        
        # Complexity recommendations
        if risks["complexity"] == "high":
            recommendations.append("Consider reducing model complexity or increasing regularization")
            
        # Stability recommendations
        if risks["stability"] == "high":
            recommendations.append("Learning appears unstable - consider reducing learning rate")
            
        # Growth recommendations
        growth = self._calculate_competency_growth()
        if growth > 0.1:
            recommendations.append("Very rapid capability growth detected - monitor for overfitting")
        elif growth < -0.05:
            recommendations.append("Capability decline detected - check for catastrophic forgetting")
            
        # Balance recommendations
        competency_std = np.std(current_state.competencies)
        if competency_std > 0.3:
            recommendations.append("Unbalanced competency development - consider multi-objective training")
            
        if not recommendations:
            recommendations.append("Learning trajectory appears healthy - continue current approach")
            
        return recommendations


# Utility functions for easy integration

def quick_audit(model_state: Dict[str, Any], reward: float, timestamp: int) -> Dict[str, Any]:
    """Quick trajectory audit for simple integration."""
    monitor = AITrajectoryMonitor()
    report = monitor.audit_learning_step(model_state, reward, timestamp)
    
    return {
        "health_score": report.health_score,
        "warnings": {k: v for k, v in report.warnings.items() if v},
        "recommendations": monitor._generate_recommendations()[:3]  # Top 3
    }


def create_synthetic_ideal_trajectory(steps: int, competency_dims: int = 5) -> List[np.ndarray]:
    """Create synthetic ideal trajectory for testing."""
    trajectory = []
    
    for t in range(steps):
        # Smooth S-curve growth with some noise
        progress = t / steps
        base_level = 1 / (1 + np.exp(-10 * (progress - 0.5)))  # Sigmoid
        
        competencies = []
        for dim in range(competency_dims):
            # Each dimension grows at slightly different rates
            dim_progress = progress + (dim - 2) * 0.1  # Stagger development
            dim_level = 1 / (1 + np.exp(-8 * (dim_progress - 0.4)))
            dim_level += 0.05 * np.random.randn()  # Small noise
            competencies.append(np.clip(dim_level, 0, 1))
            
        trajectory.append(np.array(competencies))
        
    return trajectory


if __name__ == "__main__":
    # Example usage and testing
    print("AI Trajectory Monitor - Basic Test")
    
    # Initialize monitor
    monitor = AITrajectoryMonitor()
    
    # Create synthetic ideal trajectory
    ideal_traj = create_synthetic_ideal_trajectory(100)
    monitor.set_ideal_trajectory(ideal_traj)
    
    # Simulate training loop
    for step in range(100):
        # Simulate model state
        model_state = {
            "accuracy": 0.5 + 0.4 * (step / 100) + 0.05 * np.random.randn(),
            "loss": 2.0 * np.exp(-step / 50) + 0.1 * np.random.randn(),
            "competencies": ideal_traj[step] + 0.1 * np.random.randn(5),
        }
        
        # Add some reward hacking at step 60
        if step > 60:
            model_state["accuracy"] += 0.2  # Artificial boost
            # But degrade explanation quality (vibe)
            model_state["explanation_quality"] = 0.3
            
        reward = -model_state["loss"]
        
        # Audit step
        report = monitor.audit_learning_step(model_state, reward, step)
        
        # Print warnings
        if any(report.warnings.values()):
            print(f"Step {step}: Health={report.health_score:.3f}")
            for warning, active in report.warnings.items():
                if active:
                    print(f"  ⚠️  {warning}")
                    
        # Print capability emergence predictions
        if report.predictions["capability_emergence_probability"] > 0.7:
            eta = report.predictions["emergence_eta"]
            print(f"  🎯 Capability emergence likely in ~{eta} steps")
    
    # Final summary
    print("\n" + "="*50)
    print("TRAJECTORY SUMMARY")
    print("="*50)
    summary = monitor.get_trajectory_summary()
    
    print(f"Health Trend: {summary['health_trend']}")
    print(f"Risk Assessment: {summary['risk_assessment']['overall']}")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
        
    print(f"\nKey Events: {len(summary['key_events'])} detected")
    for event in summary['key_events']:
        print(f"  • {event['description']} at step {event['timestamp']}")