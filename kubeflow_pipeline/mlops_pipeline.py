import kfp
from kfp import dsl

@dsl.container_component
def data_processing_op():
    return dsl.ContainerSpec(
        image="maskedwolf4/colorectal-cancer-prediction:latest",
        command=["uv", "run", "python", "-m", "src.data_processing"],
    )

@dsl.container_component
def model_training_op():
    return dsl.ContainerSpec(
        image="maskedwolf4/colorectal-cancer-prediction:latest",
        command=["uv", "run", "python", "-m", "src.model_training"],
    )

# Pipeline
@dsl.pipeline(
    name="mlops_pipeline",
    description="MLOps pipeline for colorectal cancer survival prediction",
)
def mlops_pipeline():
    data_processing_task = data_processing_op()
    model_training_task = model_training_op()
    model_training_task.after(data_processing_task)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(mlops_pipeline, "mlops_pipeline.yaml")
    