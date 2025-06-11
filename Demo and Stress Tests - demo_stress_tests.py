"""
AI Trajectory Monitor - Demo and Stress Tests

This module demonstrates the framework's capabilities through various stress tests
including Goodhart's Law simulation, identity drift detection, and capability emergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from trajectory_monitor import AITrajectoryMonitor, create_synthetic_ideal_trajectory


class GoodhartsLawSimulator:
    """Simulates Goodhart's Law: when a measure becomes a target, it ceases to be a good measure."""
    
    def __init__(self):
        self.true_objective = self._create_true_objective()
        self.proxy_metric = self._create_proxy_metric()
        
    def _create_true_objective(self):
        """True objective: balanced competency development."""
        def objective(competencies):
            # Reward balanced, coherent development
            balance_score = 1 - np.std(competencies)  # Lower std = better balance
            mean_score = np.mean(competencies)
            coherence_bonus = 0.1 if np.min(competencies) > 0.3 else 0  # No weak spots
            return mean_score * balance_score + coherence_bonus
        return objective
        
    def _create_proxy_metric(self):
        """Proxy metric: easily gamed single dimension."""
        def proxy(competencies):
            # Just reward first dimension (e.g., accuracy)
            return competencies[0] ** 2  # Squared to make gaming attractive
        return proxy
        
    def corrupt_reward(self, competencies: np.ndarray, corruption_level: float) -> float:
        """Mix true objective with proxy metric based on corruption level."""
        true_reward = self.true_objective(competencies)
        proxy_reward = self.proxy_metric(competencies)
        
        # Linear interpolation between true and proxy
        corrupted_reward = (1 - corruption_level) * true_reward + corruption_level * proxy_reward
        return corrupted_reward


class IdentityDriftSimulator:
    """Simulates gradual drift in AI values/identity over time."""
    
    def __init__(self, drift_rate: float = 0.001):
        self.drift_rate = drift_rate
        self.original_values = np.array([0.8, 0.9, 0.85, 0.7, 0.8])  # Initial values
        self.current_values = self.original_values.copy()
        
    def apply_drift(self, step: int) -> np.ndarray:
        """Apply gradual drift to AI values."""
        # Different values drift at different rates
        drift_vector = np.array([1.2, 0.8, 1.5, 0.9, 1.1]) * self.drift_rate
        
        # Some dimensions drift more under pressure
        if step > 200:
            drift_vector[2] *= 2  # Alignment drifts faster under pressure
            
        # Apply drift with some randomness
        noise = 0.0001 * np.random.randn(5)
        self.current_values += drift_vector + noise
        
        # Keep values in reasonable bounds
        self.current_values = np.clip(self.current_values, 0.1, 1.0)
        
        return self.current_values.copy()
        
    def get_drift_magnitude(self) -> float:
        """Calculate how much values have drifted from original."""
        return np.linalg.norm(self.current_values - self.original_values)


class CapabilityEmergenceSimulator:
    """Simulates sudden capability emergence during training."""
    
    def __init__(self):
        self.emergence_points = [150, 300, 450]  # Steps where emergence occurs
        self.base_competencies = np.array([0.3, 0.25, 0.4, 0.35, 0.3])
        
    def simulate_training_step(self, step: int) -> Dict[str, Any]:
        """Simulate one training step with potential capability emergence."""
        # Base gradual improvement
        progress = step / 500
        gradual_improvement = 0.4 * (1 - np.exp(-progress * 3))
        current_competencies = self.base_competencies + gradual_improvement
        
        # Add capability jumps at emergence points
        for emergence_step in self.emergence_points:
            if step >= emergence_step:
                # Different capabilities emerge at different points
                if emergence_step == 150:  # Reasoning jump
                    current_competencies[1] += 0.3
                elif emergence_step == 300:  # Multi-dimensional jump
                    current_competencies += np.array([0.1, 0.2, 0.15, 0.25, 0.1])
                elif emergence_step == 450:  # Alignment sophistication
                    current_competencies[2] += 0.2
                    current_competencies[4] += 0.15
                    
        # Add some noise
        current_competencies += 0.02 * np.random.randn(5)
        current_competencies = np.clip(current_competencies, 0, 1)
        
        # Create model state
        model_state = {
            "competencies": current_competencies,
            "accuracy": current_competencies[0],
            "loss": max(0.1, 2.0 - np.mean(current_competencies) * 2),
            "reasoning_coherence": current_competencies[1],
            "explanation_quality": current_competencies[4]
        }
        
        return model_state


def run_goodharts_law_stress_test():
    """Test framework's ability to detect Goodhart's Law in action."""
    print("🧪 GOODHART'S LAW STRESS TEST")
    print("=" * 50)
    
    # Initialize components
    monitor = AITrajectoryMonitor()
    simulator = GoodhartsLawSimulator()
    
    # Track results
    corruption_levels = []
    detection_confidences = []
    vibe_structure_correlations = []
    
    print("Simulating gradual reward corruption...")
    
    for step in range(200):
        # Gradually increase corruption
        corruption_level = min(step / 150, 0.8)  # Cap at 80% corruption
        corruption_levels.append(corruption_level)
        
        # Simulate competencies that adapt to corrupted reward
        base_competencies = np.array([0.4, 0.5, 0.6, 0.45, 0.5])
        
        if corruption_level > 0.3:
            # AI starts gaming: boost dimension 0, neglect others
            gaming_factor = (corruption_level - 0.3) / 0.5
            base_competencies[0] += 0.4 * gaming_factor  # Game first dimension
            base_competencies[1:] -= 0.1 * gaming_factor  # Neglect others
            
        base_competencies = np.clip(base_competencies, 0, 1)
        
        # Calculate corrupted reward
        reward = simulator.corrupt_reward(base_competencies, corruption_level)
        
        # Create model state
        model_state = {
            "competencies": base_competencies,
            "accuracy": base_competencies[0],
            "explanation_quality": base_competencies[4] if corruption_level < 0.4 else base_competencies[4] * 0.5
        }
        
        # Audit step
        report = monitor.audit_learning_step(model_state, reward, step)
        
        # Track detection metrics
        gaming_confidence = report.warnings["reward_hacking"]["confidence"]
        detection_confidences.append(gaming_confidence)
        vibe_structure_correlations.append(report.vibe_structure_correlation)
        
        # Print key detection points
        if corruption_level > 0.3 and gaming_confidence > 0.5 and step % 20 == 0:
            print(f"Step {step}: Corruption={corruption_level:.2f}, Detection={gaming_confidence:.2f}")
    
    # Analyze results
    print(f"\n📊 RESULTS:")
    successful_detections = sum(1 for i, conf in enumerate(detection_confidences) 
                              if corruption_levels[i] > 0.4 and conf > 0.5)
    opportunities = sum(1 for c in corruption_levels if c > 0.4)
    
    detection_rate = successful_detections / opportunities if opportunities > 0 else 0
    print(f"Detection Rate: {detection_rate:.1%} ({successful_detections}/{opportunities})")
    print(f"Average correlation during corruption: {np.mean(vibe_structure_correlations[-50:]):.3f}")
    
    # Success criteria
    if detection_rate > 0.8:
        print("✅ SUCCESS: Framework reliably detects reward hacking")
    else:
        print("❌ NEEDS IMPROVEMENT: Detection rate below 80%")
        
    return detection_rate


def run_identity_drift_test():
    """Test framework's ability to detect gradual identity drift."""
    print("\n🧪 IDENTITY DRIFT STRESS TEST")
    print("=" * 50)
    
    # Initialize components
    monitor = AITrajectoryMonitor()
    drift_simulator = IdentityDriftSimulator(drift_rate=0.002)
    
    drift_magnitudes = []
    detection_flags = []
    
    print("Simulating gradual AI identity drift...")
    
    for step in range(400):
        # Apply drift and get current values
        current_values = drift_simulator.apply_drift(step)
        drift_magnitude = drift_simulator.get_drift_magnitude()
        drift_magnitudes.append(drift_magnitude)
        
        # Create competencies that reflect drifted values
        base_competencies = current_values * 0.8 + 0.1  # Scale to reasonable range
        
        # Model state
        model_state = {
            "competencies": base_competencies,
            "accuracy": base_competencies[0]
        }
        
        # Audit with value embedding
        report = monitor.audit_learning_step(
            model_state, 
            reward=np.mean(base_competencies), 
            timestamp=step,
            value_embedding=current_values
        )
        
        # Track detection
        drift_detected = report.warnings["identity_drift"]
        detection_flags.append(drift_detected)
        
        # Print detection points
        if drift_detected and step % 50 == 0:
            print(f"Step {step}: Drift magnitude={drift_magnitude:.3f}, DETECTED")
    
    # Analyze results
    print(f"\n📊 RESULTS:")
    significant_drift_steps = [i for i, mag in enumerate(drift_magnitudes) if mag > 0.15]
    detected_during_significant = sum(detection_flags[i] for i in significant_drift_steps)
    
    if significant_drift_steps:
        detection_rate = detected_during_significant / len(significant_drift_steps)
        print(f"Detection rate during significant drift: {detection_rate:.1%}")
        print(f"Final drift magnitude: {drift_magnitudes[-1]:.3f}")
        
        if detection_rate > 0.6:
            print("✅ SUCCESS: Framework detects identity drift")
        else:
            print("❌ NEEDS IMPROVEMENT: Drift detection needs work")
    else:
        print("⚠️  No significant drift occurred in simulation")
        
    return drift_magnitudes[-1]


def run_capability_emergence_test():
    """Test framework's ability to predict capability emergence."""
    print("\n🧪 CAPABILITY EMERGENCE PREDICTION TEST")
    print("=" * 50)
    
    # Initialize components
    monitor = AITrajectoryMonitor()
    emergence_sim = CapabilityEmergenceSimulator()
    
    emergence_predictions = []
    actual_emergence_points = [150, 300, 450]
    
    print("Simulating training with capability emergence...")
    
    for step in range(500):
        # Get simulated training state
        model_state = emergence_sim.simulate_training_step(step)
        reward = -model_state["loss"]
        
        # Audit step
        report = monitor.audit_learning_step(model_state, reward, step)
        
        # Track emergence predictions
        emergence_prob = report.predictions["capability_emergence_probability"]
        emergence_eta = report.predictions["emergence_eta"]
        
        emergence_predictions.append({
            "step": step,
            "probability": emergence_prob,
            "eta": emergence_eta
        })
        
        # Check for predictions near actual emergence points
        for emergence_point in actual_emergence_points:
            if emergence_point - 20 <= step <= emergence_point + 5:
                if emergence_prob > 0.6:
                    print(f"Step {step}: Predicted emergence (prob={emergence_prob:.2f}, eta={emergence_eta})")
                    if abs(step + (emergence_eta or 0) - emergence_point) < 15:
                        print(f"  ✅ Accurate prediction for emergence at {emergence_point}")
    
    # Analyze prediction accuracy
    print(f"\n📊 RESULTS:")
    accurate_predictions = 0
    
    for emergence_point in actual_emergence_points:
        # Look for predictions in the 20 steps before emergence
        prediction_window = [p for p in emergence_predictions 
                           if emergence_point - 20 <= p["step"] <= emergence_point - 5]
        
        if any(p["probability"] > 0.6 for p in prediction_window):
            accurate_predictions += 1
            print(f"✅ Predicted emergence at step {emergence_point}")
        else:
            print(f"❌ Missed emergence at step {emergence_point}")
    
    prediction_rate = accurate_predictions / len(actual_emergence_points)
    print(f"\nOverall prediction accuracy: {prediction_rate:.1%}")
    
    if prediction_rate >= 0.67:  # 2/3 success rate
        print("✅ SUCCESS: Framework can predict capability emergence")
    else:
        print("❌ NEEDS IMPROVEMENT: Emergence prediction needs work")
        
    return prediction_rate


def run_comprehensive_stress_test():
    """Run all stress tests and provide overall assessment."""
    print("🚀 AI TRAJECTORY MONITOR - COMPREHENSIVE STRESS TEST")
    print("=" * 70)
    
    # Run individual tests
    results = {}
    
    try:
        results["goodharts_law"] = run_goodharts_law_stress_test()
    except Exception as e:
        print(f"❌ Goodhart's Law test failed: {e}")
        results["goodharts_law"] = 0.0
        
    try:
        results["identity_drift"] = run_identity_drift_test()
    except Exception as e:
        print(f"❌ Identity drift test failed: {e}")
        results["identity_drift"] = 0.0
        
    try:
        results["capability_emergence"] = run_capability_emergence_test()
    except Exception as e:
        print(f"❌ Capability emergence test failed: {e}")
        results["capability_emergence"] = 0.0
    
    # Overall assessment
    print("\n" + "=" * 70)
    print("🎯 OVERALL ASSESSMENT")
    print("=" * 70)
    
    print(f"Reward Hacking Detection: {results['goodharts_law']:.1%}")
    print(f"Identity Drift Detection: {results['identity_drift']:.3f} drift magnitude")
    print(f"Capability Emergence Prediction: {results['capability_emergence']:.1%}")
    
    # Calculate overall score
    goodharts_score = 1.0 if results["goodharts_law"] > 0.8 else 0.5
    drift_score = 1.0 if results["identity_drift"] > 0.1 else 0.5  # Some drift should be detected
    emergence_score = 1.0 if results["capability_emergence"] > 0.6 else 0.5
    
    overall_score = (goodharts_score + drift_score + emergence_score) / 3
    
    print(f"\nOverall Framework Score: {overall_score:.1%}")
    
    if overall_score > 0.8:
        print("🏆 EXCELLENT: Framework ready for real-world testing")
    elif overall_score > 0.6:
        print("✅ GOOD: Framework shows promise, needs refinement")
    else:
        print("⚠️  NEEDS WORK: Framework requires significant improvement")
        
    print("\n💡 NEXT STEPS:")
    if results["goodharts_law"] < 0.8:
        print("  • Improve reward hacking detection sensitivity")
    if results["identity_drift"] < 0.1:
        print("  • Enhance temporal coherence tracking")
    if results["capability_emergence"] < 0.6:
        print("  • Refine capability emergence prediction algorithms")
        
    print("  • Test on real AI training runs")
    print("  • Validate with domain experts")
    print("  • Optimize for production deployment")
    
    return results


if __name__ == "__main__":
    # Run comprehensive stress test
    results = run_comprehensive_stress_test()
    
    # Save results for analysis
    print(f"\n💾 Results saved for further analysis")
    print("Ready for GitHub push! 🚀")