"""
Example: Markov Chain Dynamic Topic Model with MixEHR-SAGE

This script demonstrates:
1. Generating patient sequential data (multiple visits per patient)
2. Initializing MixEHR-SAGE with Markov chain temporal inference
3. Performing VAE-based inference for theta_t | theta_t-1
4. Tracking and storing temporal theta evolution
5. Visualizing disease progression for individual patients
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from temporal_markov_utils import PatientSequenceGenerator, SequenceDataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mock_corpus_for_markov(num_patients=100, vocab_sizes=[500, 300]):
    """
    Create mock corpus compatible with sequential patient data
    """
    print("=" * 70)
    print("Step 1: Creating Mock Corpus for Markov Chain Model")
    print("=" * 70)
    
    # Create Document objects (one per patient, but will have multiple time steps)
    from corpus import Corpus
    
    class MockDocument:
        def __init__(self, doc_id, modality_num=2):
            self.doc_id = doc_id
            self.words_dict = [{} for _ in range(modality_num)]
            self.Cd = [0 for _ in range(modality_num)]
    
    docs = [MockDocument(i, modality_num=2) for i in range(num_patients)]
    
    # Create mock corpus
    corpus = type('Corpus', (), {
        'D': num_patients,
        'V': vocab_sizes,
        'C': [sum(vocab_sizes), sum(vocab_sizes)],  # Total word counts
        'docs': docs
    })()
    
    print(f"✓ Created corpus with {corpus.D} patients")
    print(f"✓ Vocabulary sizes: {corpus.V}")
    
    return corpus


def create_seed_matrix(vocab_sizes, num_topics=50):
    """
    Create seed topic matrix for guided modality
    """
    print("\n" + "=" * 70)
    print("Step 2: Creating Seed Topic Matrix")
    print("=" * 70)
    
    # Seeds only for first modality (e.g., ICD codes)
    seeds_matrix = torch.zeros(vocab_sizes[0], num_topics, dtype=torch.double)
    
    for k in range(num_topics):
        num_seeds = np.random.randint(5, 15)
        seed_words = np.random.choice(vocab_sizes[0], size=num_seeds, replace=False)
        seeds_matrix[seed_words, k] = 1.0
    
    print(f"✓ Created {num_topics} topics")
    print(f"✓ Average seeds per topic: {seeds_matrix.sum(0).mean():.1f}")
    
    return seeds_matrix


def demonstrate_markov_chain_inference(corpus, seeds_matrix, patient_sequences):
    """
    Demonstrate Markov chain temporal inference
    """
    print("\n" + "=" * 70)
    print("Step 3: Markov Chain Temporal Inference")
    print("=" * 70)
    
    from MixEHR_SAGE import MixEHR_SAGE
    
    # Initialize model WITH Markov chain temporal inference
    print("\nInitializing MixEHR-SAGE with Markov chain inference...")
    model = MixEHR_SAGE(
        corpus=corpus,
        seeds_topic_matrix=seeds_matrix,
        modality_list=['icd', 'medication'],
        guided_modality=0,
        batch_size=50,
        enable_temporal=True,  # Enable Markov chain inference
        num_time_steps=10  # Max time steps per patient
    )
    
    print(f"✓ Model initialized")
    print(f"✓ Temporal storage: {model.theta_temporal.shape}")
    print(f"✓ VAE hidden size: {model.theta_hidden_size}")
    print(f"✓ Markov transition variance: {model.transition_variance}")
    
    # Create data loader
    loader = SequenceDataLoader(patient_sequences, corpus.V)
    
    # Perform inference for a few example patients
    print("\nPerforming VAE inference for patient sequences...")
    example_patients = list(patient_sequences.keys())[:5]
    
    results = {}
    for patient_id in example_patients:
        try:
            # Get patient sequence data
            seq_data = loader.get_patient_sequence_data(patient_id)
            
            print(f"\nPatient {patient_id}: {len(seq_data)} visits")
            
            # Perform variational inference
            theta_samples, mu_theta, logvar_theta = model.infer_theta_variational(
                seq_data, patient_id
            )
            
            print(f"  ✓ Inferred theta sequence: {theta_samples.shape}")
            
            # Compute KL divergence for Markov chain
            kl_loss = model.compute_markov_chain_kl(theta_samples, mu_theta, logvar_theta)
            print(f"  ✓ Markov chain KL divergence: {kl_loss:.3f}")
            
            # Store results
            results[patient_id] = {
                'theta_samples': theta_samples.cpu().numpy(),
                'num_visits': len(seq_data),
                'kl_loss': kl_loss.item()
            }
            
            # Show topic distribution for first and last visit
            print(f"  Top 5 topics at visit 1: {torch.topk(theta_samples[0], 5)[1].cpu().numpy()}")
            if len(seq_data) > 1:
                print(f"  Top 5 topics at visit {len(seq_data)}: {torch.topk(theta_samples[-1], 5)[1].cpu().numpy()}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    return model, results


def visualize_patient_trajectories(model, results, save_path='./markov_trajectories.png'):
    """
    Visualize how theta evolves over time for patients
    """
    print("\n" + "=" * 70)
    print("Step 4: Visualizing Patient Trajectories")
    print("=" * 70)
    
    if not results:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Theta evolution for first patient (first 5 topics)
    ax = axes[0, 0]
    patient_id = list(results.keys())[0]
    theta = results[patient_id]['theta_samples']
    num_visits = results[patient_id]['num_visits']
    
    for k in range(min(5, theta.shape[1])):
        ax.plot(range(num_visits), theta[:, k], marker='o', label=f'Topic {k+1}')
    ax.set_xlabel('Visit Number')
    ax.set_ylabel('Topic Probability')
    ax.set_title(f'Patient {patient_id}: Theta Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Topic diversity over time (entropy)
    ax = axes[0, 1]
    for patient_id, data in results.items():
        theta = data['theta_samples']
        entropy = -np.sum(theta * np.log(theta + 1e-10), axis=1)
        ax.plot(range(data['num_visits']), entropy, marker='o', label=f'Patient {patient_id}')
    ax.set_xlabel('Visit Number')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Topic Diversity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Dominant topic per visit for one patient
    ax = axes[1, 0]
    patient_id = list(results.keys())[0]
    theta = results[patient_id]['theta_samples']
    dominant_topics = np.argmax(theta, axis=1)
    num_visits = results[patient_id]['num_visits']
    
    ax.bar(range(num_visits), dominant_topics, color='steelblue', alpha=0.7)
    ax.set_xlabel('Visit Number')
    ax.set_ylabel('Dominant Topic ID')
    ax.set_title(f'Patient {patient_id}: Dominant Topic Per Visit')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: KL divergence comparison
    ax = axes[1, 1]
    patient_ids = list(results.keys())
    kl_values = [results[pid]['kl_loss'] for pid in patient_ids]
    num_visits = [results[pid]['num_visits'] for pid in patient_ids]
    
    scatter = ax.scatter(num_visits, kl_values, s=100, alpha=0.6, c=num_visits, cmap='viridis')
    ax.set_xlabel('Number of Visits')
    ax.set_ylabel('Total KL Divergence')
    ax.set_title('KL Divergence vs. Number of Visits')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Visits')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")
    plt.close()


def main():
    """
    Run complete Markov chain demonstration
    """
    print("\n" + "=" * 70)
    print("MixEHR-SAGE: Markov Chain Dynamic Topic Model")
    print("=" * 70)
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    num_patients = 50
    vocab_sizes = [500, 300]  # ICD, Medication
    num_topics = 50
    
    # Generate patient sequences
    print("\n" + "=" * 70)
    print("Generating Patient Sequential Data")
    print("=" * 70)
    
    gen = PatientSequenceGenerator(vocab_sizes, num_modalities=2)
    patient_sequences, metadata = gen.generate_synthetic_patient_sequences(
        num_patients=num_patients,
        min_visits=3,
        max_visits=8,
        progression_strength=0.4  # Topics evolve over visits
    )
    
    print(f"✓ Generated {len(patient_sequences)} patients")
    avg_visits = np.mean([len(visits) for visits in patient_sequences.values()])
    print(f"✓ Average visits per patient: {avg_visits:.1f}")
    
    # Create corpus and seeds
    corpus = create_mock_corpus_for_markov(num_patients, vocab_sizes)
    seeds_matrix = create_seed_matrix(vocab_sizes, num_topics)
    
    # Perform Markov chain inference
    model, results = demonstrate_markov_chain_inference(corpus, seeds_matrix, patient_sequences)
    
    # Visualize results
    visualize_patient_trajectories(model, results)
    
    # Save temporal theta
    if model.enable_temporal:
        save_path = './temporal_theta_markov.pt'
        model.save_temporal_theta(save_path)
        print(f"\n✓ Saved temporal theta to {save_path}")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print("\nKey Achievements:")
    print("1. ✓ Implemented Markov chain: p(theta_t | theta_t-1)")
    print("2. ✓ VAE encoder: q(theta_t | X_1...t-1, theta_t-1)")
    print("3. ✓ Stored temporal theta for each patient and time step")
    print("4. ✓ Computed KL divergence for Markov transitions")
    print("5. ✓ Visualized patient disease progression trajectories")
    print("\nDifferences from Population-Level Approach:")
    print("- Individual patient trajectories (not age groups)")
    print("- Sequential visits with temporal dependencies")
    print("- Theta evolves per patient (not fixed alpha per time bin)")
    print("- Suitable for personalized disease progression modeling")


if __name__ == "__main__":
    main()
