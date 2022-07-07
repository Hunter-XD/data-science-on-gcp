# Copyright 2017-2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from datetime import datetime
import tensorflow as tf

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from kfp.v2 import compiler, dsl

ENDPOINT_NAME = 'flights'


def train_custom_model(data_set, timestamp, develop_mode, cpu_only_mode, tf_version, extra_args=None):
    # Set up training and deployment infra

    if cpu_only_mode:
        train_image = (
            f'us-docker.pkg.dev/vertex-ai/training/tf-cpu.{tf_version}:latest'
        )

    else:
        train_image = (
            f"us-docker.pkg.dev/vertex-ai/training/tf-gpu.{tf_version}:latest"
        )

    deploy_image = (
        f'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.{tf_version}:latest'
    )

    # train
    model_display_name = f'{ENDPOINT_NAME}-{timestamp}'
    job = aiplatform.CustomTrainingJob(
        display_name=f'train-{model_display_name}',
        script_path="model.py",
        container_uri=train_image,
        requirements=['cloudml-hypertune'],
        model_serving_container_image_uri=deploy_image,
    )

    model_args = [
        '--bucket', BUCKET,
    ]
    if develop_mode:
        model_args += ['--develop']
    if extra_args:
        model_args += extra_args

    return (
        job.run(
            dataset=data_set,
            # See https://googleapis.dev/python/aiplatform/latest/aiplatform.html#
            predefined_split_column_name='data_split',
            model_display_name=model_display_name,
            args=model_args,
            replica_count=1,
            machine_type='n1-standard-4',
            sync=develop_mode,
        )
        if cpu_only_mode
        else job.run(
            dataset=data_set,
            # See https://googleapis.dev/python/aiplatform/latest/aiplatform.html#
            predefined_split_column_name='data_split',
            model_display_name=model_display_name,
            args=model_args,
            replica_count=1,
            machine_type='n1-standard-4',
            # See https://cloud.google.com/vertex-ai/docs/general/locations#accelerators
            accelerator_type=aip.AcceleratorType.NVIDIA_TESLA_T4.name,
            accelerator_count=1,
            sync=develop_mode,
        )
    )


def train_automl_model(data_set, timestamp, develop_mode):
    # train
    model_display_name = f'{ENDPOINT_NAME}-{timestamp}'
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f'train-{model_display_name}',
        optimization_prediction_type='classification',
    )

    return job.run(
        dataset=data_set,
        predefined_split_column_name='data_split',
        target_column='ontime',
        model_display_name=model_display_name,
        budget_milli_node_hours=(300 if develop_mode else 2000),
        disable_early_stopping=False,
        export_evaluated_data_items=True,
        export_evaluated_data_items_bigquery_destination_uri=f'{PROJECT}:dsongcp.ch9_automl_evaluated',
        export_evaluated_data_items_override_destination=True,
        sync=develop_mode,
    )


def do_hyperparameter_tuning(data_set, timestamp, develop_mode, cpu_only_mode, tf_version):
    # Vertex AI services require regional API endpoints.
    if cpu_only_mode:
        train_image = (
            f'us-docker.pkg.dev/vertex-ai/training/tf-cpu.{tf_version}:latest'
        )

    else: 
        train_image = (
            f"us-docker.pkg.dev/vertex-ai/training/tf-gpu.{tf_version}:latest"
        )


    # a single trial job
    model_display_name = f'{ENDPOINT_NAME}-{timestamp}'
    if cpu_only_mode:
        trial_job = aiplatform.CustomJob.from_local_script(
            display_name=f'train-{model_display_name}',
            script_path="model.py",
            container_uri=train_image,
            args=[
                '--bucket',
                BUCKET,
                '--skip_full_eval',  # no need to evaluate on test data set
                '--num_epochs',
                '10',
                '--num_examples',
                '500000',  # 1/10 actual size to finish faster
            ],
            requirements=['cloudml-hypertune'],
            replica_count=1,
            machine_type='n1-standard-4',
        )

    else:
        trial_job = aiplatform.CustomJob.from_local_script(
            display_name=f'train-{model_display_name}',
            script_path="model.py",
            container_uri=train_image,
            args=[
                '--bucket',
                BUCKET,
                '--skip_full_eval',  # no need to evaluate on test data set
                '--num_epochs',
                '10',
                '--num_examples',
                '500000',  # 1/10 actual size to finish faster
            ],
            requirements=['cloudml-hypertune'],
            replica_count=1,
            machine_type='n1-standard-4',
            accelerator_type=aip.AcceleratorType.NVIDIA_TESLA_T4.name,
            accelerator_count=1,
        )


    # the tuning job
    hparam_job = aiplatform.HyperparameterTuningJob(
        display_name=f'hparam-{model_display_name}',
        custom_job=trial_job,
        metric_spec={'val_rmse': 'minimize'},
        parameter_spec={
            "train_batch_size": hpt.IntegerParameterSpec(
                min=16, max=256, scale='log'
            ),
            "nbuckets": hpt.IntegerParameterSpec(
                min=5, max=10, scale='linear'
            ),
            "dnn_hidden_units": hpt.CategoricalParameterSpec(
                values=["64,16", "64,16,4", "64,64,64,8", "256,64,16"]
            ),
        },
        max_trial_count=2 if develop_mode else NUM_HPARAM_TRIALS,
        parallel_trial_count=2,
        search_algorithm=None,
    )


    hparam_job.run(sync=True)  # has to finish before we can get trials.

    # get the parameters corresponding to the best trial
    best = sorted(hparam_job.trials, key=lambda x: x.final_measurement.metrics[0].value)[0]
    logging.info(f'Best trial: {best}')
    best_params = []
    for param in best.parameters:
        best_params.append(f'--{param.parameter_id}')

        if param.parameter_id in ["train_batch_size", "nbuckets"]:
            # hparam returns 10.0 even though it's an integer param. so round it.
            # but CustomTrainingJob makes integer args into floats. so make it a string
            best_params.append(str(int(round(param.value))))
        else:
            # string or float parameters
            best_params.append(param.value)

    # run the best trial to completion
    logging.info(f'Launching full training job with {best_params}')
    return train_custom_model(data_set, timestamp, develop_mode, cpu_only_mode, tf_version, extra_args=best_params)


@dsl.pipeline(name="flights-ch9-pipeline",
              description="ds-on-gcp ch9 flights pipeline"
)
def main():
    aiplatform.init(
        project=PROJECT, location=REGION, staging_bucket=f'gs://{BUCKET}'
    )


    # create data set
    all_files = tf.io.gfile.glob(f'gs://{BUCKET}/ch9/data/all*.csv')
    logging.info(f"Training on {all_files}")
    data_set = aiplatform.TabularDataset.create(
        display_name=f'data-{ENDPOINT_NAME}', gcs_source=all_files
    )

    if TF_VERSION is not None:
        tf_version = TF_VERSION.replace(".", "-")
    else:
        tf_version = f'2-{tf.__version__[2:3]}'

    # train
    if AUTOML:
        model = train_automl_model(data_set, TIMESTAMP, DEVELOP_MODE)
    elif NUM_HPARAM_TRIALS > 1:
        model = do_hyperparameter_tuning(data_set, TIMESTAMP, DEVELOP_MODE, CPU_ONLY_MODE, tf_version)
    else:
        model = train_custom_model(data_set, TIMESTAMP, DEVELOP_MODE, CPU_ONLY_MODE, tf_version)

    # create endpoint if it doesn't already exist
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_NAME}"',
        order_by='create_time desc',
        project=PROJECT,
        location=REGION,
    )

    if len(endpoints) > 0:
        endpoint = endpoints[0]  # most recently created
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=ENDPOINT_NAME, project=PROJECT, location=REGION,
            sync=DEVELOP_MODE
        )

    # deploy
    model.deploy(
        endpoint=endpoint,
        traffic_split={"0": 100},
        machine_type='n1-standard-2',
        min_replica_count=1,
        max_replica_count=1,
        sync=DEVELOP_MODE
    )

    if DEVELOP_MODE:
        model.wait()


def run_pipeline():
    compiler.Compiler().compile(pipeline_func=main, package_path='flights_pipeline.json')

    job = aip.PipelineJob(
        display_name=f"{ENDPOINT_NAME}-pipeline",
        template_path=f"{ENDPOINT_NAME}_pipeline.json",
        pipeline_root=f"{BUCKET}/pipeline_root/intro",
        enable_caching=False,
    )


    job.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bucket',
        help='Data will be read from gs://BUCKET/ch9/data and checkpoints will be in gs://BUCKET/ch9/trained_model',
        required=True
    )
    parser.add_argument(
        '--region',
        help='Where to run the trainer',
        default='us-central1'
    )
    parser.add_argument(
        '--project',
        help='Project to be billed',
        required=True
    )
    parser.add_argument(
        '--develop',
        help='Train on a small subset in development',
        dest='develop',
        action='store_true')
    parser.set_defaults(develop=False)
    parser.add_argument(
        '--automl',
        help='Train an AutoML Table, instead of using model.py',
        dest='automl',
        action='store_true')
    parser.set_defaults(automl=False)
    parser.add_argument(
        '--num_hparam_trials',
        help='Number of hyperparameter trials. 0/1 means no hyperparam. Ignored if --automl is set.',
        type=int,
        default=0)
    parser.add_argument(
        '--pipeline',
        help='Run as pipeline',
        dest='pipeline',
        action='store_true')
    parser.add_argument(
        '--cpuonly',
        help='Run without GPU',
        dest='cpuonly',
        action='store_true')
    parser.set_defaults(cpuonly=False)
    parser.add_argument(
        '--tfversion',
        help='TensorFlow version to use'
    )

    # parse args
    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args().__dict__
    BUCKET = args['bucket']
    PROJECT = args['project']
    REGION = args['region']
    DEVELOP_MODE = args['develop']
    CPU_ONLY_MODE = args['cpuonly']
    TF_VERSION = args['tfversion']    
    AUTOML = args['automl']
    NUM_HPARAM_TRIALS = args['num_hparam_trials']
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    if args['pipeline']:
        run_pipeline()
    else:
        main()