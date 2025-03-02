from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logging import logger
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig
import sys

if __name__ == "__main__":
    try:
        logger.info("Data ingestion started")
        dataingestionConfig = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
        data_ingestion = DataIngestion(data_ingestion_config=dataingestionConfig)
        dataIngestionArtifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed")
        print("-------------------------")

        print(dataIngestionArtifact)

        
        logger.info("Data validation started")
        data_validation_config = DataValidationConfig(training_pipeline_config=TrainingPipelineConfig())
        datavalidation = DataValidation(data_ingestion_artifact=dataIngestionArtifact,data_validation_config=data_validation_config)
        data_validation_artifact = datavalidation.initiate_data_validation()
        logger.info("Data validation completed")
        print("-------------------------")
        print(data_validation_artifact)
        
        logger.info("Data transformation started")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=TrainingPipelineConfig())
        datatransformation = DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=data_transformation_config)
        data_transformation_artifact = datatransformation.initiate_data_transformation()
        logger.info("Data transformation completed")
        print("-------------------------")

        print(data_transformation_artifact)
        
        

        
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
