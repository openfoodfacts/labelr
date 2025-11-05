import datetime

import typer
from google.cloud import batch_v1

app = typer.Typer()


@app.command()
def train_object_detection(
    wandb_project: str = typer.Option(
        "train-yolo", help="The Weights & Biases project name."
    ),
    wandb_api_key: str = typer.Option(..., envvar="WANDB_API_KEY"),
    hf_token: str = typer.Option(
        ...,
        help="The Hugging Face token, used to push the trained model to Hugging Face Hub.",
    ),
    run_name: str = typer.Option(..., help="A name for the training run."),
    hf_repo_id: str = typer.Option(
        ..., help="The Hugging Face dataset repository ID to use to train."
    ),
    hf_trained_model_repo_id: str = typer.Option(
        ..., help="The Hugging Face repository ID where to push the trained model."
    ),
    epochs: int = typer.Option(100, help="Number of training epochs."),
    imgsz: int = typer.Option(640, help="Size of the image during training."),
    batch_size: int = typer.Option(64, help="Batch size for training."),
):
    """Train an object detection model."""
    env_variables = {
        "HF_REPO_ID": hf_repo_id,
        "HF_TRAINED_MODEL_REPO_ID": hf_trained_model_repo_id,
        "HF_TOKEN": hf_token,
        "WANDB_PROJECT": wandb_project,
        "RUN_NAME": run_name,
        "WANDB_API_KEY": wandb_api_key,
        "EPOCHS": str(epochs),
        "IMGSZ": str(imgsz),
        "BATCH_SIZE": str(batch_size),
        "USE_AWS_IMAGE_CACHE": "False",
    }
    job_name = "train-yolo-job"
    job_name = job_name + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    job = launch_job(
        job_name=job_name,
        container_image_uri="europe-west9-docker.pkg.dev/robotoff/gcf-artifacts/train-yolo",
        env_variables=env_variables,
    )
    typer.echo("Job launched")
    typer.echo(job)


def launch_job(
    job_name: str = typer.Argument(
        ...,
        help="The name of the Google Batch job that will be created. "
        "It needs to be unique for each project and region pair.",
    ),
    container_image_uri: str = typer.Argument(
        ..., help="The URI of the container image that will be run as part of the job."
    ),
    commands: str | None = None,
    env_variables: dict[str, str] | None = None,
    entrypoint: str | None = None,
    cpu_milli: int = 4000,  # in milli-CPU units (4000 = 4 CPUs). This means the task requires 4 whole CPUs.
    memory_mib: int = 16000,
    boot_disk_mib: int = 100000,
    max_retry_count: int = 1,
    max_run_duration: str = "86400s",  # 24 hours
    task_count: int = 1,
    accelerators_type: str = "nvidia-tesla-t4",
    machine_type: str = "n1-standard-8",
    google_project_id: str = "robotoff",
    accelerators_count: int = 1,
    region: str = "europe-west4",
    install_gpu_drivers: bool = True,
) -> batch_v1.Job:
    """This method creates a Batch Job on GCP.

    Sources:
    * https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/batch/create
    * https://cloud.google.com/python/docs/reference/batch/latest/google.cloud.batch_v1.types  # noqa

    :param google_batch_launch_config: Config to run a job on Google Batch.
    :param batch_job_config: Config to run a specific job on Google Batch.
    :return: Batch job information.

    Returns:
        Batch job information.
    """
    client = batch_v1.BatchServiceClient()

    # Define what will be done as part of the job.
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = container_image_uri
    runnable.container.entrypoint = entrypoint  # type: ignore
    runnable.container.commands = commands

    # Jobs can be divided into tasks. In this case, we have only one task.
    task = batch_v1.TaskSpec()
    task.runnables = [runnable]

    # Environment variables.
    envable = batch_v1.Environment()
    envable.variables = env_variables or {}
    task.environment = envable

    # We can specify what resources are requested by each task.
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = cpu_milli
    resources.memory_mib = memory_mib
    resources.boot_disk_mib = boot_disk_mib  # type: ignore
    task.compute_resource = resources

    task.max_retry_count = max_retry_count
    task.max_run_duration = max_run_duration  # type: ignore

    # Tasks are grouped inside a job using TaskGroups.
    group = batch_v1.TaskGroup()
    group.task_count = task_count  # type: ignore
    group.task_spec = task

    # Policies are used to define on what kind of virtual machines the tasks
    # will run on.
    policy = batch_v1.AllocationPolicy.InstancePolicy()
    # See list of machine types here:
    # https://docs.cloud.google.com/compute/docs/gpus#l4-gpus
    policy.machine_type = machine_type

    accelerator = batch_v1.AllocationPolicy.Accelerator()
    accelerator.type_ = accelerators_type
    accelerator.count = accelerators_count

    policy.accelerators = [accelerator]
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    instances.install_gpu_drivers = install_gpu_drivers
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]

    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy
    # We use Cloud Logging as it's an out of the box available option
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING  # type: ignore

    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    create_request.job_id = job_name
    # The job's parent is the region in which the job will run
    create_request.parent = f"projects/{google_project_id}/locations/{region}"

    return client.create_job(create_request)
