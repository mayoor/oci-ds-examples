import argparse
import os
import shutil
from os import cpu_count, path

import ads
import pandas as pd
from ads.common.model_export_util import prepare_generic_model
from ads.model.deployment import ModelDeployer, ModelDeploymentProperties
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark.sql.types import DateType, IntegerType, StructType


def load_data(spark, data_path):

    fraud_anomaly = spark.read.csv(
        data_path,
        header=True,
        inferSchema=True,
    )
    print(fraud_anomaly.columns)

    print(fraud_anomaly.dtypes)

    fraud_anomaly.createOrReplaceTempView("FRAUD_DATA")

    print(spark.sql("select col01,col011  from FRAUD_DATA limit 10").show())

    fraud_anomaly.groupBy("anomalous").count().show()
    spark.sql("select distinct anomalous from FRAUD_DATA").show()

    train, test = fraud_anomaly.randomSplit([0.8, 0.2], seed=42)

    train.groupBy("anomalous").count().show()
    test.groupBy("anomalous").count().show()

    return train, test, fraud_anomaly


def create_model(data_column, label_column):
    vec_assembler = VectorAssembler(inputCols=data_column, outputCol="features")
    gbt_model = GBTClassifier(maxIter=5, maxDepth=2, labelCol=label_column, seed=42)
    ml_pipeline = Pipeline(stages=[vec_assembler, gbt_model])
    return ml_pipeline


def train_model(ml_pipeline, train, test, label_column):
    model = ml_pipeline.fit(train)

    predictions = model.transform(test)

    predictions.select("prediction").show(10)

    predictions.groupBy("prediction", label_column).count().show()

    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_column, predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(
        "F-1 Score:{}".format(
            evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        )
    )
    print("Test Error = %g" % (1.0 - accuracy))
    return model


def save_model(model, location):
    sparkmodel = location
    model.write().overwrite().save(sparkmodel)


def test_load_model(model_location, test, label_column):
    from pyspark.ml import Pipeline, PipelineModel

    model2 = PipelineModel.load(model_location)
    newpreds = model2.transform(test)
    newpreds.groupBy("prediction", label_column).count().show()


def save_model_to_catalog(
    model_artifact_location,
    spark_model_location,
    storage_options,
    compartment_id,
    project_id,
):
    sparkmodel = spark_model_location

    modelartifact = prepare_generic_model(
        model_artifact_location,
        data_science_env=False,
        inference_conda_env="",
        force_overwrite=True,
        ignore_deployment_error=True,
    )


    shutil.copytree(sparkmodel, f"{model_artifact_location}/{sparkmodel}")
    with open(f"{model_artifact_location}/runtime.yaml", "w") as f:
        f.write(
            """# Model runtime environment
MODEL_ARTIFACT_VERSION: '3.0'
MODEL_DEPLOYMENT:
  INFERENCE_CONDA_ENV:
    INFERENCE_ENV_PATH: oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/PySpark
      3.0 and Data Flow/2.0/pyspark30_p37_cpu_v2
    INFERENCE_ENV_SLUG: pyspark30_p37_cpu_v2
    INFERENCE_ENV_TYPE: data_science
    INFERENCE_PYTHON_VERSION: 3.7.10    
        """
        )

    with open(f"{model_artifact_location}/__init__.py", "w") as f:
        f.write(
            """
        """
        )

    with open(f"{model_artifact_location}/score.py", "w") as f:
        f.write(
            """
import json
import os
from functools import lru_cache
from pyspark.sql import SparkSession
import pandas as pd


model_name = 'attrition_gbt_model'
from pyspark.ml import Pipeline, PipelineModel




spark = SparkSession.builder.appName("My GBT Classifier").getOrCreate()

@lru_cache(maxsize=10)
def load_model(model_file_name=model_name):
    
    return PipelineModel.load('attrition_gbt_model')



def predict(data, model=load_model()):

    columns = ['Charges', 'col01', 'col010', 'col011', 'col012', 'col013', 'col014', 'col015', 'col016', 'col017', 'col018', 'col019', 'col02', 'col020', 'col021', 'col022', 'col023', 'col024', 'col025', 'col026', 'col027', 'col028', 'col029', 'col03', 'col030', 'col031', 'col032', 'col033', 'col04', 'col05', 'col06', 'col07', 'col08', 'col09', 'login_count']
    features = data['data']
    pdf = pd.DataFrame(data=[features], columns=columns)
    sparkdf = spark.createDataFrame(pdf) 
    print(features)
    predictions = model.transform(sparkdf)
    return {"prediction":predictions.select("prediction").toPandas()["prediction"].tolist()}    
    
        """
        )

    mc_model = modelartifact.save(
        project_id=project_id,
        compartment_id=compartment_id,
        display_name="GBT Classfier for Fraud Anomaly Detection",
        description="GBT Classfier for Fraud Anomaly Detection",
        ignore_pending_changes=True,
        timeout=100,
        ignore_introspection=True,
    )

    return mc_model


def deploy_model(
    model_id, name, project_id, compartment_id, logId, logGroupId, authinfo
):
    model_deployment_properties = (
        ModelDeploymentProperties(model_id)
        .with_prop("display_name", name)
        .with_prop("project_id", project_id)
        .with_prop("compartment_id", compartment_id)
        .with_logging_configuration(logGroupId, logId, logGroupId, logId)
        .with_instance_configuration(
            config={
                "INSTANCE_SHAPE": "VM.Standard2.1",
                "INSTANCE_COUNT": "1",
                "bandwidth_mbps": 10,
            }
        )
    )

    deployer = ModelDeployer(config=authinfo)
    deployment = deployer.deploy(model_deployment_properties)
    return deployment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="DEFAULT", help="oci config profile")
    parser.add_argument(
        "--auth",
        default="resource_principal",
        help="auth type",
        type=str,
        choices=["api_key", "resource_principal"],
    )
    parser.add_argument(
        "--project_ocid",
        default=os.environ.get("PROJECT_OCID"),
        help="auth type",
        type=str,
    )
    parser.add_argument(
        "--compartment_ocid",
        default=os.environ.get("NB_SESSION_COMPARTMENT_OCID"),
        help="auth type",
        type=str,
    )
    parser.add_argument("--loggroup_ocid", help="Log Group OCID", type=str)
    parser.add_argument("--log_ocid", help="Logging OCID", type=str)
    parser.add_argument("--data", type=str, help="oci uri path in the format oci://{bucket}@{namespace}/{datapath} to oracle test fraud dataset")
    args = parser.parse_args()

    profile = args.profile
    ads.set_auth(args.auth, profile)

    storage_options = (
        {"config": "~/.oci/config", "profile": profile}
        if args.auth == "api_key"
        else {}
    )

    model_deployment_authinfo = (
        {"auth": "api_key", "oci_config_profile": profile}
        if args.auth == "api_key"
        else {"auth": "resource_principal"}
    )

    PROJECT_OCID = args.project_ocid
    COMPARTMENT_OCID = args.compartment_ocid
    logGroupId = args.loggroup_ocid
    logId = args.log_ocid

    data_path = args.data

    spark_session = (
        SparkSession.builder.appName("Python Spark SQL basic example")
        .config("spark.driver.cores", str(max(1, cpu_count() - 1)))
        .config("spark.executor.cores", str(max(1, cpu_count() - 1)))
        .getOrCreate()
    )

    train, test, fraud_anomaly = load_data(spark_session, data_path)

    label_column = "anomalous"
    data_column = [col for col in fraud_anomaly.columns if col != label_column]

    model_artifact_location = "mygbtmodel"
    model_location = "attrition_gbt_model"

    pipeline = create_model(label_column=label_column, data_column=data_column)
    model = train_model(pipeline, train, test, label_column)
    save_model(model, model_location)

    test_load_model(model_location, test, label_column)

    print("Saving spark model to Model Catalog...")
    mc_model = save_model_to_catalog(
        model_artifact_location,
        model_location,
        storage_options,
        project_id=PROJECT_OCID,
        compartment_id=COMPARTMENT_OCID,
    )

    print(f"Save complete. Model Id is {mc_model.id}")

    print(f"Deploying model....")

    md = deploy_model(
        mc_model.id,
        "Fraud Anomaly GBT Classfifier",
        PROJECT_OCID,
        COMPARTMENT_OCID,
        logId,
        logGroupId,
        model_deployment_authinfo,
    )
    
    print(f"Model Deployed successfully..")

    print(f"Testing the end point...")
    df = pd.read_csv(
        data_path,
        storage_options=storage_options,
    )
    data_column = [col for col in df.columns if col != "anomalous"]

    test_data = df[data_column].iloc[0].tolist()

    prediction_endpoint_test_data = {"data":test_data}

    print(f"Model Prediction: {md.predict(prediction_endpoint_test_data)}")