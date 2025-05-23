from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import *
import yaml

def create_mlops_workflow():
    """Create automated MLOps workflow in Databricks"""
    
    w = WorkspaceClient()
    
    # Define the job configuration
    job_config = {
        "name": "MLOps-Regression-Pipeline",
        "tags": {"team": "data-science", "project": "mlops"},
        "tasks": [
            {
                "task_key": "data-validation",
                "notebook_task": {
                    "notebook_path": "/Workspace/Repos/your-repo/mlops-project/notebooks/01_data_exploration",
                    "source": "WORKSPACE"
                },
                "job_cluster_key": "ml-cluster",
                "timeout_seconds": 3600
            },
            {
                "task_key": "training-pipeline",
                "depends_on": [{"task_key": "data-validation"}],
                "python_wheel_task": {
                    "package_name": "mlops_project",
                    "entry_point": "training_pipeline"
                },
                "job_cluster_key": "ml-cluster",
                "timeout_seconds": 7200
            },
            {
                "task_key": "model-validation",
                "depends_on": [{"task_key": "training-pipeline"}],
                "notebook_task": {
                    "notebook_path": "/Workspace/Repos/your-repo/mlops-project/notebooks/04_model_evaluation",
                    "source": "WORKSPACE"
                },
                "job_cluster_key": "ml-cluster",
                "timeout_seconds": 1800
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "ml-cluster",
                "new_cluster": {
                    "spark_version": "13.3.x-ml-scala2.12",
                    "node_type_id": "i3.xlarge",
                    "num_workers": 2,
                    "spark_conf": {
                        "spark.databricks.delta.preview.enabled": "true"
                    },
                    "azure_attributes": {
                        "availability": "SPOT_AZURE",
                        "first_on_demand": 1,
                        "spot_bid_max_price": -1
                    }
                }
            }
        ],
        "schedule": {
            "quartz_cron_expression": "0 0 2 * * ?",  # Daily at 2 AM
            "timezone_id": "UTC"
        },
        "email_notifications": {
            "on_success": ["your-email@company.com"],
            "on_failure": ["your-email@company.com"]
        },
        "max_concurrent_runs": 1
    }
    
    # Create the job
    job = w.jobs.create(**job_config)
    print(f"Created job with ID: {job.job_id}")
    
    return job.job_id

# File-based trigger for new data
def setup_file_trigger():
    """Setup file-based trigger for automatic pipeline execution"""
    
    trigger_code = """
    # File trigger notebook
    import dbutils
    import time
    
    # Check for new data files
    trigger_path = "/mnt/datalake/trigger/"
    
    try:
        files = dbutils.fs.ls(trigger_path)
        if files:
            print(f"Found {len(files)} trigger files")
            
            # Run training pipeline
            dbutils.notebook.run(
                "/Workspace/Repos/your-repo/mlops-project/pipelines/training_pipeline",
                timeout_seconds=7200
            )
            
            # Clean up trigger files
            for file in files:
                dbutils.fs.rm(file.path)
                
            print("Pipeline execution completed")
        else:
            print("No trigger files found")
            
    except Exception as e:
        print(f"Error: {e}")
    """
    
    return trigger_code

if __name__ == "__main__":
    job_id = create_mlops_workflow()
    print(f"MLOps workflow created with job ID: {job_id}")