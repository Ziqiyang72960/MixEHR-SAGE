"""
Example: Using Temporal Features in MixEHR-SAGE

This script demonstrates how to:
1. Generate synthetic temporal EHR data
2. Initialize MixEHR-SAGE with temporal inference enabled
3. Perform temporal inference with LSTM
4. Visualize temporal topic evolution
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from temporal_utils import TemporalDataGenerator, TemporalSequencePreprocessor
from corpus import Corpus, Document
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_synthetic_temporal_corpus(num_patients=1000, vocab_size=500, num_time_steps=10):
    """
    Create a synthetic corpus with temporal information
    """
    print("=" * 60)
    print("Step 1: Generating Synthetic Temporal Data")
    print("=" * 60)
    
    # Initialize temporal data generator
    gen = TemporalDataGenerator(
        num_time_steps=num_time_steps,
        min_age=0,
        max_age=100
    )
    
    # Generate synthetic patients with ages
    documents, patient_ages = gen.generate_synthetic_temporal_data(
        num_patients=num_patients,
        vocab_size=vocab_size,
        words_per_patient=50,
        trend_strength=0.5  # Strong temporal trend
    )
    
    print(f"✓ Generated {len(documents)} patients")
    print(f"✓ Age range: {min(patient_ages.values()):.1f} - {max(patient_ages.values()):.1f} years")
    print(f"✓ Vocabulary size: {vocab_size}")
    
    # Create time bins
    time_bins = gen.create_temporal_corpus_from_ages(type('Corpus', (), {'D': len(documents)})(), patient_ages)
    
    print(f"\nTime bin distribution:")
    for t in range(num_time_steps):
        count = len(time_bins.get(t, []))
        print(f"  Bin {t} ({gen.age_bins[t]:.0f}-{gen.age_bins[t+1]:.0f} years): {count} patients")
    
    return documents, patient_ages, time_bins, gen


def create_mock_corpus_and_seeds(documents, vocab_size, num_topics=50):
    """
    Create mock corpus and seed topic matrix for demonstration
    """
    print("\n" + "=" * 60)
    print("Step 2: Creating Corpus and Seed Topics")
    print("=" * 60)
    
    # Create Document objects
    docs = []
    for doc_id, word_freq in documents:
        doc = Document()
        doc.doc_id = doc_id
        doc.words_dict = {0: word_freq}  # Modality 0 (e.g., ICD codes)
        docs.append(doc)
    
    # Create mock corpus
    corpus = type('Corpus', (), {
        'D': len(docs),
        'V': [vocab_size],
        'C': [sum(sum(d.words_dict[0].values()) for d in docs)],
        'docs': docs
    })()
    
    print(f"✓ Created corpus with {corpus.D} documents")
    print(f"✓ Total word count: {corpus.C[0]}")
    
    # Create seed topic matrix (V x K)
    # Each topic has 5-15 seed words
    seeds_matrix = torch.zeros(vocab_size, num_topics, dtype=torch.double)
    for k in range(num_topics):
        num_seeds = np.random.randint(5, 15)
        seed_words = np.random.choice(vocab_size, size=num_seeds, replace=False)
        seeds_matrix[seed_words, k] = 1.0
    
    print(f"✓ Created {num_topics} topics with seed words")
    print(f"✓ Average seeds per topic: {seeds_matrix.sum(0).mean():.1f}")
    
    return corpus, seeds_matrix


def demonstrate_temporal_inference(corpus, seeds_matrix, time_bins, gen, patient_ages):
    """
    Demonstrate temporal inference with MixEHR-SAGE
    """
    print("\n" + "=" * 60)
    print("Step 3: Temporal Inference with LSTM")
    print("=" * 60)
    
    # Import here to avoid circular dependencies
    from MixEHR_SAGE import MixEHR_SAGE
    
    # Initialize model WITH temporal inference
    print("\nInitializing MixEHR-SAGE with temporal inference enabled...")
    model = MixEHR_SAGE(
        corpus=corpus,
        seeds_topic_matrix=seeds_matrix,
        modality_list=['icd'],
        guided_modality=0,
        batch_size=100,
        enable_temporal=True,  # Enable temporal inference
        num_time_steps=gen.T
    )
    
    print(f"✓ Model initialized with {model.T} time steps")
    print(f"✓ LSTM hidden size: {model.eta_hidden_size}")
    print(f"✓ LSTM layers: {model.eta_nlayers}")
    
    # Generate temporal word distributions
    print("\nAggregating word distributions by time...")
    time_word_dist = gen.aggregate_word_distributions_by_time(corpus, time_bins, modality=0)
    print(f"✓ Created temporal sequence: {time_word_dist.shape}")
    
    # Perform variational inference for eta
    print("\nPerforming LSTM-based variational inference...")
    try:
        eta_samples, mu_eta, logvar_eta = model.infer_eta_variational(time_word_dist)
        
        print(f"✓ Sampled eta: {eta_samples.shape}")
        print(f"✓ Mean eta range: [{mu_eta.min():.3f}, {mu_eta.max():.3f}]")
        print(f"✓ Log-variance range: [{logvar_eta.min():.3f}, {logvar_eta.max():.3f}]")
        
        # Compute alpha (Dirichlet parameters) via softplus
        alpha = model.alpha_softplus_act()
        print(f"✓ Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
        
        # Compute temporal KL divergence
        kl_loss = model.compute_temporal_kl(mu_eta, logvar_eta)
        print(f"✓ Temporal KL divergence: {kl_loss:.3f}")
        
        return eta_samples, mu_eta, logvar_eta, alpha
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def visualize_temporal_evolution(eta_samples, alpha, num_time_steps, save_path='./temporal_viz.png'):
    """
    Visualize how topics evolve over time
    """
    if eta_samples is None:
        print("\nSkipping visualization due to inference error")
        return
    
    print("\n" + "=" * 60)
    print("Step 4: Visualizing Temporal Evolution")
    print("=" * 60)
    
    # Convert to numpy for plotting
    eta_np = eta_samples.detach().cpu().numpy()
    alpha_np = alpha.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Eta evolution for first 5 topics
    ax = axes[0, 0]
    for k in range(min(5, eta_np.shape[1])):
        ax.plot(range(num_time_steps), eta_np[:, k], marker='o', label=f'Topic {k+1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Eta Value')
    ax.set_title('Temporal Evolution of Eta (First 5 Topics)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Alpha evolution for first 5 topics
    ax = axes[0, 1]
    for k in range(min(5, alpha_np.shape[1])):
        ax.plot(range(num_time_steps), alpha_np[:, k], marker='s', label=f'Topic {k+1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Alpha Value (Softplus of Eta)')
    ax.set_title('Temporal Evolution of Alpha (First 5 Topics)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of all eta values
    ax = axes[1, 0]
    im = ax.imshow(eta_np.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Topic')
    ax.set_title('Eta Heatmap (All Topics)')
    plt.colorbar(im, ax=ax, label='Eta Value')
    
    # Plot 4: Mean and variance across topics
    ax = axes[1, 1]
    mean_eta = eta_np.mean(axis=1)
    std_eta = eta_np.std(axis=1)
    ax.plot(range(num_time_steps), mean_eta, marker='o', label='Mean', linewidth=2)
    ax.fill_between(range(num_time_steps), 
                    mean_eta - std_eta, 
                    mean_eta + std_eta, 
                    alpha=0.3, label='±1 Std Dev')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Eta Value')
    ax.set_title('Average Eta Across All Topics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")
    plt.close()


def main():
    """
    Run complete demonstration
    """
    print("\n" + "=" * 60)
    print("MixEHR-SAGE Temporal Inference Demonstration")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    num_patients = 1000
    vocab_size = 500
    num_topics = 50
    num_time_steps = 10
    
    # Step 1: Generate data
    documents, patient_ages, time_bins, gen = create_synthetic_temporal_corpus(
        num_patients=num_patients,
        vocab_size=vocab_size,
        num_time_steps=num_time_steps
    )
    
    # Step 2: Create corpus and seeds
    corpus, seeds_matrix = create_mock_corpus_and_seeds(
        documents, vocab_size, num_topics
    )
    
    # Step 3: Temporal inference
    eta_samples, mu_eta, logvar_eta, alpha = demonstrate_temporal_inference(
        corpus, seeds_matrix, time_bins, gen, patient_ages
    )
    
    # Step 4: Visualization
    visualize_temporal_evolution(eta_samples, alpha, num_time_steps)
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Temporal inference captures time-varying topic distributions")
    print("2. LSTM learns smooth transitions between time steps")
    print("3. Alpha (via softplus) ensures positive Dirichlet parameters")
    print("4. KL divergence regularizes temporal dynamics")
    print("\nNext Steps:")
    print("- Use real EHR data with actual patient ages")
    print("- Integrate with full inference loop")
    print("- Tune hyperparameters (learning rate, KL weight, etc.)")
    print("- Evaluate temporal topic coherence")


if __name__ == "__main__":
    main()
