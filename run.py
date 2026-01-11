from src.pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Run pipeline with SMOTE strategy
    run = training_pipeline(strategy='smote')
    
    # Get results from the pipeline run
    print("\nPipeline completed successfully!")
    print(f"Pipeline run ID: {run.name}")


        # Baseline (no sampling)
#             'random_over',    # Random oversampling
#             'random_under',   # Random undersampling
#             'smote',          # Synthetic Minority Oversampling
#             'adasyn',         # Adaptive Synthetic Sampling
#             'smote_tomek',    # SMOTE + Tomek Links
#             'smote_enn'