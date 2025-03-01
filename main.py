from networksecurity.components.dats_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logging import logger
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        dataingestionConfig = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
        data_ingestion = DataIngestion(data_ingestion_config=dataingestionConfig)
        dataIngestionArtifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed")
        print(dataIngestionArtifact)
        
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
