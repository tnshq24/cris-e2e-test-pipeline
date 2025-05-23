from src.model.predictor import ModelPredictor
from src.data_ingestion.data_loader import DataLoader
from src.feature_store.feature_manager import FeatureManager
import pandas as pd
from pyspark.sql.functions import *

class InferencePipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.predictor = ModelPredictor(config_path)
        self.data_loader = DataLoader(config_path)
        self.feature_manager = FeatureManager(config_path)
    
    def run_batch_inference(self, input_data_path: str, output_path: str):
        """Run batch inference on new data"""
        
        print("🔮 Starting batch inference pipeline...")
        
        # Load new data
        new_data = self.data_loader.load_raw_data(input_data_path)
        print(f"📥 Loaded {new_data.count()} rows for inference")
        
        # Get features from Feature Store
        primary_keys = new_data.select("id").rdd.map(lambda row: row[0]).collect()
        feature_data = self.feature_manager.get_latest_features(primary_keys)
        
        # Make predictions
        predictions = self.predictor.predict_batch(feature_data)
        
        # Add predictions to DataFrame
        predictions_df = pd.DataFrame({
            'id': primary_keys,
            'prediction': predictions,
            'prediction_timestamp': pd.Timestamp.now()
        })
        
        # Convert to Spark DataFrame and save
        spark_predictions = spark.createDataFrame(predictions_df)
        spark_predictions.write.mode("overwrite").parquet(output_path)
        
        print(f"💾 Saved predictions to: {output_path}")
        return predictions_df
    
    def run_streaming_inference(self, input_stream_path: str, output_stream_path: str):
        """Run streaming inference"""
        
        # Read streaming data
        streaming_df = (spark
                       .readStream
                       .format("delta")
                       .load(input_stream_path))
        
        # Define prediction function
        def predict_batch_udf(batch_df, batch_id):
            if batch_df.count() > 0:
                # Convert to Pandas for prediction
                pandas_df = batch_df.toPandas()
                
                # Make predictions
                predictions = self.predictor.predict_batch(pandas_df)
                
                # Add predictions
                pandas_df['prediction'] = predictions
                pandas_df['batch_id'] = batch_id
                pandas_df['prediction_timestamp'] = pd.Timestamp.now()
                
                # Convert back to Spark and write
                result_df = spark.createDataFrame(pandas_df)
                (result_df
                 .write
                 .format("delta")
                 .mode("append")
                 .save(output_stream_path))
        
        # Start streaming query
        query = (streaming_df
                .writeStream
                .foreachBatch(predict_batch_udf)
                .outputMode("append")
                .trigger(processingTime='30 seconds')
                .start())
        
        return query