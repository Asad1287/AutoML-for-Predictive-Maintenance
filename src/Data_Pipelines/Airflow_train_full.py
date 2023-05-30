
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from CONSTANTS import *
from airflow.utils.dates import days_ago
from src.Data_Modeling.model_search import model_search

from MongoHandler import *
# Import your functions from the scripts
from Data_Ingestion.Batch_Processing.data_ingestion import process_data
from Data_Modeling.model_search import xgboost_optimize_hyperparameters
from Data_Modeling.model_creation import xgboost_model



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
   
}

dag = DAG(
    'model_training_dag',
    default_args=default_args,
    description='DAG for training the model',
    schedule_interval='@daily',
    start_date=days_ago(2)

)

t1 = PythonOperator(
    task_id='xgboost_optimize_hyperparameters',
    python_callable=process_data,
    op_kwargs={
        'client': MongoDBHandler(f"mongodb+srv://root12345:{PASSWORD}@cluster1.b03tix4.mongodb.net/","predictive_maintenance","predictive_maintenance"),
        'Train': True,
    },
    dag=dag,
)
#col_names:List[str],parquet_file:str,save_parquet_file_path:str
t2 = PythonOperator(
    task_id='dim_reduction',
    python_callable=dim_reduction_processing,
    op_kwargs={
      'col_names':col_names,
      'parquet_file':parquet_file,
      'save_parquet_file_path':save_parquet_file_path

      
    },
    dag=dag,
)