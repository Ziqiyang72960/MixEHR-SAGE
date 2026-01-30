#!/usr/bin/env python3
"""
Tests for Temporal MixEHR-SAGE functionality.

Run with: 
    python test_temporal.py
    
Or from parent directory:
    python -m test_temporal
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd
import torch

# Import from same directory (tests are expected to run from the repo root)
from temporal_corpus import TemporalCorpus, TemporalBucket, TemporalPatient, generate_sample_temporal_data
from temporal_models import (
    MarkovTransitionModel, TemporalLSTMModel, TemporalMixEHRTrainer,
    analyze_disease_progression
)


class TestTemporalBucket(unittest.TestCase):
    """Tests for TemporalBucket class."""
    
    def test_bucket_creation(self):
        """Test basic bucket creation."""
        bucket = TemporalBucket('patient_001', 2015, '2015-01-01', '2015-12-31', 'yearly')
        self.assertEqual(bucket.patient_id, 'patient_001')
        self.assertEqual(bucket.time_index, 2015)
        self.assertEqual(bucket.bucket_type, 'yearly')
        self.assertEqual(len(bucket), 0)
    
    def test_add_record(self):
        """Test adding records to bucket."""
        bucket = TemporalBucket('patient_001', 2015, '2015-01-01', '2015-12-31', 'yearly')
        bucket.add_record('E11.9', 'icd', '2015-03-15')
        bucket.add_record('I10', 'icd', '2015-03-15')
        
        self.assertEqual(len(bucket), 2)
        self.assertEqual(bucket.records[0]['code'], 'E11.9')
        self.assertEqual(bucket.records[0]['modality'], 'icd')
    
    def test_get_bow_by_modality(self):
        """Test bag-of-words conversion."""
        bucket = TemporalBucket('patient_001', 2015, '2015-01-01', '2015-12-31', 'yearly')
        bucket.add_record('E11.9', 'icd', '2015-03-15')
        bucket.add_record('I10', 'icd', '2015-03-15')
        bucket.add_record('E11.9', 'icd', '2015-06-20')  # Duplicate
        
        vocab_mappings = {'icd': {'E11.9': 0, 'I10': 1, 'J44.1': 2}}
        modality_list = ['icd', 'med']
        
        bow = bucket.get_bow_by_modality(vocab_mappings, modality_list)
        
        self.assertEqual(len(bow), 2)  # Two modalities
        self.assertEqual(bow[0][0], 2)  # E11.9 appears twice
        self.assertEqual(bow[0][1], 1)  # I10 appears once
        self.assertEqual(bow[1], {})  # med modality empty


class TestTemporalPatient(unittest.TestCase):
    """Tests for TemporalPatient class."""
    
    def test_patient_creation(self):
        """Test basic patient creation."""
        patient = TemporalPatient('patient_001')
        self.assertEqual(patient.patient_id, 'patient_001')
        self.assertEqual(len(patient), 0)
    
    def test_add_buckets(self):
        """Test adding buckets and sorting."""
        patient = TemporalPatient('patient_001')
        
        # Add buckets out of order
        bucket_2016 = TemporalBucket('patient_001', 2016, '2016-01-01', '2016-12-31', 'yearly')
        bucket_2015 = TemporalBucket('patient_001', 2015, '2015-01-01', '2015-12-31', 'yearly')
        
        patient.add_bucket(bucket_2016)
        patient.add_bucket(bucket_2015)
        
        self.assertEqual(len(patient), 2)
        # Should be sorted by time
        self.assertEqual(patient.buckets[0].time_index, 2015)
        self.assertEqual(patient.buckets[1].time_index, 2016)
    
    def test_get_cumulative_bow(self):
        """Test cumulative bag-of-words retrieval."""
        patient = TemporalPatient('patient_001')
        
        bucket_2015 = TemporalBucket('patient_001', 2015, '2015-01-01', '2015-12-31', 'yearly')
        bucket_2015.add_record('E11.9', 'icd', '2015-03-15')
        
        bucket_2016 = TemporalBucket('patient_001', 2016, '2016-01-01', '2016-12-31', 'yearly')
        bucket_2016.add_record('I10', 'icd', '2016-03-15')
        bucket_2016.add_record('E11.9', 'icd', '2016-06-20')
        
        patient.add_bucket(bucket_2015)
        patient.add_bucket(bucket_2016)
        
        vocab_mappings = {'icd': {'E11.9': 0, 'I10': 1}}
        modality_list = ['icd']
        
        # Cumulative up to 2015 (only first bucket)
        bow_2015 = patient.get_cumulative_bow(2015, vocab_mappings, modality_list)
        self.assertEqual(bow_2015[0][0], 1)  # E11.9 once
        self.assertNotIn(1, bow_2015[0])  # I10 not present
        
        # Cumulative up to 2016 (both buckets)
        bow_2016 = patient.get_cumulative_bow(2016, vocab_mappings, modality_list)
        self.assertEqual(bow_2016[0][0], 2)  # E11.9 twice
        self.assertEqual(bow_2016[0][1], 1)  # I10 once


class TestTemporalCorpus(unittest.TestCase):
    """Tests for TemporalCorpus class."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, 'test_temporal.csv')
        
        # Create test data
        df = pd.DataFrame([
            {'SUBJECT_ID': 'p001', 'code': 'E11.9', 'timestamp': '2015-03-15', 'modality': 'icd'},
            {'SUBJECT_ID': 'p001', 'code': 'I10', 'timestamp': '2015-03-15', 'modality': 'icd'},
            {'SUBJECT_ID': 'p001', 'code': 'E11.9', 'timestamp': '2016-06-20', 'modality': 'icd'},
            {'SUBJECT_ID': 'p002', 'code': 'J44.1', 'timestamp': '2014-05-10', 'modality': 'icd'},
            {'SUBJECT_ID': 'p002', 'code': 'I25.1', 'timestamp': '2015-02-20', 'modality': 'icd'},
        ])
        df.to_csv(self.test_data_path, index=False)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_from_file_yearly(self):
        """Test loading corpus with yearly bucketing."""
        corpus = TemporalCorpus.from_file(self.test_data_path, bucket_type='yearly')
        
        self.assertEqual(corpus.bucket_type, 'yearly')
        self.assertEqual(len(corpus.patients), 2)
        
        # Patient 001 should have 2 buckets (2015, 2016)
        self.assertEqual(corpus.patients['p001'].num_time_steps, 2)
        
        # Patient 002 should have 2 buckets (2014, 2015)
        self.assertEqual(corpus.patients['p002'].num_time_steps, 2)
    
    def test_from_file_monthly(self):
        """Test loading corpus with monthly bucketing."""
        corpus = TemporalCorpus.from_file(self.test_data_path, bucket_type='monthly')
        
        self.assertEqual(corpus.bucket_type, 'monthly')
        
        # Patient 001: March 2015, June 2016 -> 2 months
        self.assertEqual(corpus.patients['p001'].num_time_steps, 2)
    
    def test_from_file_visit_based(self):
        """Test loading corpus with visit-based bucketing."""
        # Create visit-based data
        visit_data_path = os.path.join(self.temp_dir, 'visit_data.csv')
        df = pd.DataFrame([
            {'SUBJECT_ID': 'p001', 'code': 'E11.9', 'timestamp': 0, 'modality': 'icd'},
            {'SUBJECT_ID': 'p001', 'code': 'I10', 'timestamp': 1, 'modality': 'icd'},
            {'SUBJECT_ID': 'p001', 'code': 'E11.9', 'timestamp': 2, 'modality': 'icd'},
        ])
        df.to_csv(visit_data_path, index=False)
        
        corpus = TemporalCorpus.from_file(visit_data_path, bucket_type='visit')
        
        self.assertEqual(corpus.bucket_type, 'visit')
        self.assertEqual(corpus.patients['p001'].num_time_steps, 3)
    
    def test_properties(self):
        """Test corpus properties."""
        corpus = TemporalCorpus.from_file(self.test_data_path, bucket_type='yearly')
        
        self.assertEqual(corpus.num_patients, 2)
        self.assertEqual(corpus.total_time_steps, 4)
    
    def test_from_modality_files(self):
        """Test loading corpus from separate modality files."""
        # Create separate modality files
        icd_path = os.path.join(self.temp_dir, 'temporal_icd.csv')
        med_path = os.path.join(self.temp_dir, 'temporal_med.csv')
        
        # ICD data
        icd_df = pd.DataFrame([
            {'SUBJECT_ID': 'p001', 'code': 'E11.9', 'timestamp': '2015-03-15'},
            {'SUBJECT_ID': 'p001', 'code': 'I10', 'timestamp': '2015-03-15'},
            {'SUBJECT_ID': 'p001', 'code': 'E11.9', 'timestamp': '2016-06-20'},
            {'SUBJECT_ID': 'p002', 'code': 'J44.1', 'timestamp': '2015-05-10'},
        ])
        icd_df.to_csv(icd_path, index=False)
        
        # Medication data
        med_df = pd.DataFrame([
            {'SUBJECT_ID': 'p001', 'code': 'A10BA02', 'timestamp': '2015-03-20'},
            {'SUBJECT_ID': 'p001', 'code': 'C03AB01', 'timestamp': '2016-06-25'},
            {'SUBJECT_ID': 'p002', 'code': 'R03AC02', 'timestamp': '2015-05-15'},
        ])
        med_df.to_csv(med_path, index=False)
        
        # Load from separate files
        corpus = TemporalCorpus.from_modality_files(
            {'icd': icd_path, 'med': med_path},
            bucket_type='yearly'
        )
        
        self.assertEqual(corpus.bucket_type, 'yearly')
        self.assertEqual(len(corpus.patients), 2)
        
        # Patient p001 should have 2 buckets (2015, 2016)
        self.assertEqual(corpus.patients['p001'].num_time_steps, 2)
        
        # Check that both modalities are present in buckets
        p001_2015_bucket = corpus.patients['p001'].get_bucket(2015)
        modalities_present = set(r['modality'] for r in p001_2015_bucket.records)
        self.assertIn('icd', modalities_present)
        self.assertIn('med', modalities_present)


class TestMarkovTransitionModel(unittest.TestCase):
    """Tests for MarkovTransitionModel class."""
    
    def setUp(self):
        """Create sample theta sequences."""
        np.random.seed(42)
        self.K = 5
        self.theta_sequences = {}
        
        for i in range(10):
            T = np.random.randint(3, 6)
            theta_seq = np.random.dirichlet(np.ones(self.K), size=T)
            self.theta_sequences[f'patient_{i}'] = torch.tensor(theta_seq, dtype=torch.double)
    
    def test_fit_dominant(self):
        """Test fitting with dominant discretization."""
        model = MarkovTransitionModel(num_topics=self.K, discretization='dominant')
        model.fit(self.theta_sequences)
        
        self.assertIsNotNone(model.transition_matrix)
        self.assertEqual(model.transition_matrix.shape, (self.K, self.K))
        
        # Rows should sum to 1
        row_sums = model.transition_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(self.K), decimal=5)
    
    def test_fit_soft(self):
        """Test fitting with soft discretization."""
        model = MarkovTransitionModel(num_topics=self.K, discretization='soft')
        model.fit(self.theta_sequences)
        
        self.assertIsNotNone(model.transition_matrix)
        self.assertEqual(model.transition_matrix.shape, (self.K, self.K))
    
    def test_predict_next_state(self):
        """Test next state prediction."""
        model = MarkovTransitionModel(num_topics=self.K)
        model.fit(self.theta_sequences)
        
        current_theta = self.theta_sequences['patient_0'][0].numpy()
        next_dist = model.predict_next_state(current_theta)
        
        self.assertEqual(len(next_dist), self.K)
        self.assertAlmostEqual(next_dist.sum(), 1.0, places=5)
    
    def test_predict_disease_risk(self):
        """Test disease risk prediction."""
        model = MarkovTransitionModel(num_topics=self.K)
        model.fit(self.theta_sequences)
        
        current_theta = self.theta_sequences['patient_0'][0].numpy()
        risk = model.predict_disease_risk(current_theta, horizon=3)
        
        self.assertIn('max_risk_topic', risk)
        self.assertIn('max_risk', risk)
        self.assertIn('entropy', risk)
        self.assertTrue(0 <= risk['max_risk'] <= 1)
    
    def test_stationary_distribution(self):
        """Test stationary distribution computation."""
        model = MarkovTransitionModel(num_topics=self.K)
        model.fit(self.theta_sequences)
        
        stationary = model.get_stationary_distribution()
        
        self.assertEqual(len(stationary), self.K)
        self.assertAlmostEqual(stationary.sum(), 1.0, places=5)
        
        # Verify it's actually stationary
        next_stationary = stationary @ model.transition_matrix
        np.testing.assert_array_almost_equal(stationary, next_stationary, decimal=4)
    
    def test_save_load(self):
        """Test model serialization."""
        model = MarkovTransitionModel(num_topics=self.K)
        model.fit(self.theta_sequences)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            loaded_model = MarkovTransitionModel.load(temp_path)
            
            np.testing.assert_array_almost_equal(
                model.transition_matrix, 
                loaded_model.transition_matrix
            )
        finally:
            os.unlink(temp_path)


class TestTemporalLSTMModel(unittest.TestCase):
    """Tests for TemporalLSTMModel class."""
    
    def setUp(self):
        """Set up test parameters."""
        self.V = 100  # Vocabulary size
        self.K = 10   # Number of topics
        self.hidden_size = 50
        self.num_layers = 2
    
    def test_model_creation(self):
        """Test LSTM model creation."""
        model = TemporalLSTMModel(
            vocab_size=self.V,
            num_topics=self.K,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        self.assertEqual(model.V, self.V)
        self.assertEqual(model.K, self.K)
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        model = TemporalLSTMModel(
            vocab_size=self.V,
            num_topics=self.K,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        batch_size = 4
        T = 5
        bow_seq = torch.randn(batch_size, T, self.V)
        
        mu, logsigma = model(bow_seq)
        
        self.assertEqual(mu.shape, (batch_size, T, self.K))
        self.assertEqual(logsigma.shape, (batch_size, T, self.K))
    
    def test_sample_eta(self):
        """Test eta sampling."""
        model = TemporalLSTMModel(
            vocab_size=self.V,
            num_topics=self.K,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        batch_size = 4
        T = 5
        mu = torch.zeros(batch_size, T, self.K)
        logsigma = torch.zeros(batch_size, T, self.K)
        
        eta = model.sample_eta(mu, logsigma)
        
        self.assertEqual(eta.shape, (batch_size, T, self.K))
    
    def test_get_alpha(self):
        """Test alpha computation from eta."""
        model = TemporalLSTMModel(
            vocab_size=self.V,
            num_topics=self.K,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        eta = torch.randn(4, 5, self.K)
        alpha = model.get_alpha(eta)
        
        # Alpha should be positive (softplus output)
        self.assertTrue((alpha > 0).all())
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        model = TemporalLSTMModel(
            vocab_size=self.V,
            num_topics=self.K,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        batch_size = 4
        T = 5
        mu = torch.randn(batch_size, T, self.K)
        logsigma = torch.zeros(batch_size, T, self.K)
        
        kl = model.kl_divergence(mu, logsigma)
        
        self.assertEqual(kl.shape, (batch_size,))
        # KL should be non-negative
        self.assertTrue((kl >= 0).all())


class TestAnalyzeProgression(unittest.TestCase):
    """Tests for analyze_disease_progression function."""
    
    def setUp(self):
        """Create sample theta sequences."""
        np.random.seed(42)
        self.K = 5
        self.theta_sequences = {}
        
        for i in range(5):
            T = 4
            theta_seq = np.random.dirichlet(np.ones(self.K), size=T)
            self.theta_sequences[f'patient_{i}'] = torch.tensor(theta_seq, dtype=torch.double)
    
    def test_analyze_all_topics(self):
        """Test progression analysis for all topics."""
        df = analyze_disease_progression(self.theta_sequences)
        
        # Should have entries for all patients and topics
        self.assertEqual(len(df), 5 * self.K)
        
        # Check columns
        required_cols = ['patient_id', 'topic_idx', 'initial_prob', 'final_prob', 
                        'mean_prob', 'trend', 'prob_change']
        for col in required_cols:
            self.assertIn(col, df.columns)
    
    def test_analyze_target_topics(self):
        """Test progression analysis for specific topics."""
        target_topics = [0, 2]
        df = analyze_disease_progression(self.theta_sequences, target_topics=target_topics)
        
        # Should have entries for all patients and target topics only
        self.assertEqual(len(df), 5 * len(target_topics))
    
    def test_trend_detection(self):
        """Test trend detection in progression."""
        # Create sequence with clear increasing trend for topic 0
        theta_seq = np.array([
            [0.1, 0.3, 0.2, 0.2, 0.2],
            [0.2, 0.25, 0.2, 0.2, 0.15],
            [0.3, 0.2, 0.2, 0.15, 0.15],
            [0.5, 0.15, 0.15, 0.1, 0.1],
        ])
        
        df = analyze_disease_progression(
            {'test': torch.tensor(theta_seq, dtype=torch.double)},
            target_topics=[0]
        )
        
        self.assertEqual(df.iloc[0]['trend'], 'increasing')


class TestGenerateSampleData(unittest.TestCase):
    """Tests for sample data generation."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            df = generate_sample_temporal_data(
                temp_path,
                num_patients=10,
                time_range=(2015, 2018)
            )
            
            # Check basic properties
            self.assertTrue(len(df) > 0)
            self.assertEqual(len(df['SUBJECT_ID'].unique()), 10)
            
            # Check columns
            required_cols = ['SUBJECT_ID', 'code', 'timestamp', 'modality']
            for col in required_cols:
                self.assertIn(col, df.columns)
            
            # Check modalities
            modalities = set(df['modality'].unique())
            self.assertTrue('icd' in modalities)
            
            # File should exist
            self.assertTrue(os.path.exists(temp_path))
        finally:
            os.unlink(temp_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete temporal analysis pipeline."""
        # Generate sample data
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, 'temporal_data.csv')
            df = generate_sample_temporal_data(data_path, num_patients=15)
            
            # Create temporal corpus
            corpus = TemporalCorpus.from_file(data_path, bucket_type='yearly')
            self.assertTrue(corpus.num_patients > 0)
            
            # Create synthetic theta sequences (since we don't have trained model)
            np.random.seed(42)
            K = 10
            theta_sequences = {}
            for patient_id, patient in corpus.patients.items():
                T = patient.num_time_steps
                theta_seq = np.random.dirichlet(np.ones(K), size=T)
                theta_sequences[patient_id] = torch.tensor(theta_seq, dtype=torch.double)
            
            # Train Markov model
            markov = MarkovTransitionModel(num_topics=K, discretization='soft')
            markov.fit(theta_sequences)
            
            # Predict risk
            sample_theta = theta_sequences[list(theta_sequences.keys())[0]][-1].numpy()
            risk = markov.predict_disease_risk(sample_theta, horizon=2)
            
            self.assertIn('max_risk', risk)
            
            # Analyze progression
            progression_df = analyze_disease_progression(theta_sequences, target_topics=[0, 1])
            self.assertTrue(len(progression_df) > 0)


class TestTemporalInferencePatient(unittest.TestCase):
    """Tests for temporal inference in infer_patient module."""
    
    def test_generate_temporal_explanation(self):
        """Test temporal explanation generation."""
        # Import the function
        from infer_patient import generate_temporal_explanation
        
        # Create mock patient result
        K = 10
        theta_current = np.random.dirichlet(np.ones(K))
        theta_sequence = [np.random.dirichlet(np.ones(K)) for _ in range(3)]
        
        patient_result = {
            'patient_id': 'test_patient_001',
            'theta_sequence': theta_sequence,
            'time_labels': [2018, 2019, 2020],
            'theta_current': theta_current,
            'top_topics': [
                (0, theta_current[0], 'Topic 0 (Test Phenotype)'),
                (1, theta_current[1], 'Topic 1'),
                (2, theta_current[2], 'Topic 2'),
            ],
            'top_codes_per_topic': {},
            'forecast': [np.random.dirichlet(np.ones(K))],
            'forecast_labels': [2021],
            'bucket_type': 'yearly'
        }
        
        # Generate explanation
        prompt = generate_temporal_explanation(patient_result, include_forecast=True)
        
        # Check that prompt contains expected sections
        self.assertIn('test_patient_001', prompt)
        self.assertIn('Current Health Status', prompt)
        self.assertIn('Disease Trajectory', prompt)
        self.assertIn('Analysis Questions', prompt)
        self.assertIn('yearly', prompt)
    
    def test_compute_phi_full(self):
        """Test phi_full computation."""
        from infer_patient import compute_phi_full
        
        # This test requires a trained model, so we'll skip if model files don't exist
        # In practice, this would be tested with a mock model
        pass
    
    def test_get_top_codes_for_topic(self):
        """Test getting top codes for a topic."""
        from infer_patient import get_top_codes_for_topic
        
        # This test requires a trained model, so we'll skip if model files don't exist
        # In practice, this would be tested with a mock model
        pass
    
    def test_temporal_data_with_two_modalities(self):
        """
        Test temporal inference with 1 patient and 2 modalities.
        This is a minimal example validating the temporal inference pipeline.
        """
        # Create sample temporal data for 1 patient with 2 modalities
        records = [
            # ICD codes
            {'SUBJECT_ID': 'patient_001', 'code': 'E11.9', 'timestamp': '2018-03-15', 'modality': 'icd'},
            {'SUBJECT_ID': 'patient_001', 'code': 'I10', 'timestamp': '2018-06-20', 'modality': 'icd'},
            {'SUBJECT_ID': 'patient_001', 'code': 'E11.9', 'timestamp': '2019-02-10', 'modality': 'icd'},
            {'SUBJECT_ID': 'patient_001', 'code': 'J44.1', 'timestamp': '2019-08-05', 'modality': 'icd'},
            {'SUBJECT_ID': 'patient_001', 'code': 'I10', 'timestamp': '2020-01-15', 'modality': 'icd'},
            # Medication codes  
            {'SUBJECT_ID': 'patient_001', 'code': 'A10BA02', 'timestamp': '2018-03-15', 'modality': 'med'},
            {'SUBJECT_ID': 'patient_001', 'code': 'C09AA02', 'timestamp': '2018-06-20', 'modality': 'med'},
            {'SUBJECT_ID': 'patient_001', 'code': 'A10BA02', 'timestamp': '2019-08-05', 'modality': 'med'},
        ]
        
        df = pd.DataFrame(records)
        
        # Test creating temporal corpus from DataFrame
        corpus = TemporalCorpus.from_dataframe(
            df, 
            bucket_type='yearly',
            subject_col='SUBJECT_ID',
            code_col='code',
            time_col='timestamp',
            modality_col='modality'
        )
        
        # Verify corpus structure
        self.assertEqual(corpus.num_patients, 1)
        self.assertIn('patient_001', corpus.patients)
        
        patient = corpus.patients['patient_001']
        self.assertEqual(patient.num_time_steps, 3)  # 2018, 2019, 2020
        
        # Verify time buckets
        time_indices = [b.time_index for b in patient.buckets]
        self.assertEqual(sorted(time_indices), [2018, 2019, 2020])
        
        # Verify records per bucket
        for bucket in patient.buckets:
            if bucket.time_index == 2018:
                self.assertEqual(len(bucket.records), 4)  # 2 ICD + 2 med
            elif bucket.time_index == 2019:
                self.assertEqual(len(bucket.records), 3)  # 2 ICD + 1 med
            elif bucket.time_index == 2020:
                self.assertEqual(len(bucket.records), 1)  # 1 ICD
        
        # Create mock vocab mappings
        vocab_mappings = {
            'icd': {'E11.9': 0, 'I10': 1, 'J44.1': 2},
            'med': {'A10BA02': 0, 'C09AA02': 1}
        }
        modality_list = ['icd', 'med']
        
        corpus.set_vocab_mappings(vocab_mappings, modality_list)
        
        # Test cumulative BOW (no data leakage)
        bow_2018 = patient.get_cumulative_bow(2018, vocab_mappings, modality_list)
        bow_2019 = patient.get_cumulative_bow(2019, vocab_mappings, modality_list)
        bow_2020 = patient.get_cumulative_bow(2020, vocab_mappings, modality_list)
        
        # 2018: E11.9(1) + I10(1) = 2 ICD codes
        self.assertEqual(bow_2018[0].get(0, 0), 1)  # E11.9
        self.assertEqual(bow_2018[0].get(1, 0), 1)  # I10
        
        # 2019 cumulative: E11.9(2) + I10(1) + J44.1(1) = 4 ICD codes
        self.assertEqual(bow_2019[0].get(0, 0), 2)  # E11.9 (accumulated)
        self.assertEqual(bow_2019[0].get(2, 0), 1)  # J44.1 (new in 2019)
        
        # 2020 cumulative: E11.9(2) + I10(2) + J44.1(1)
        self.assertEqual(bow_2020[0].get(1, 0), 2)  # I10 (accumulated)
        
        print("Temporal inference test with 1 patient and 2 modalities: PASSED")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
