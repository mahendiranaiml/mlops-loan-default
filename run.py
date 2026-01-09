from src.pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Run pipeline with SMOTE strategy
    run = training_pipeline(strategy='smote_enn')
    
    # Get results from the pipeline run
    print("\nPipeline completed successfully!")
    print(f"Pipeline run ID: {run.name}")
