# AutoML-for-Predictive-Maintenance

![alt text](https://github.com/Asad1287/AutoML-for-Predictive-Maintenance/blob/master/pexels-kateryna-babaieva-2760241.jpg)

Predictive Maintenance, the practice of forecasting machinery malfunctions and maintenance issues before they occur, is a key differentiator in today's competitive environment. It is an integral part of modern manufacturing, aiming to minimize downtime, optimize equipment lifetimes, and reduce maintenance costs, all while ensuring a smooth and efficient operation.

The solution aims to build a highly scablale, and effective data engineering and machine learning solution for solving predictive mainteannce solutions and high workloads, given hundreads and thousands of real world machines and assets tracked per second.


The solution archieture is composed of 4 main layers and will be run using the make file. 

Data Ingestion/Streaming
Data Processing 
Data Modelling 
Data Inference and API

The solution is based on batch processing and stream processing , for predictive maintenance application. 

For streaming processing, the repo uses MQTT, a publish-subscribe-based messaging protocol, and Kafka, a distributed event streaming platform, allowing your business to respond to insights gleaned from your data in real-time.Furthermore, the system integrates Spark Streaming, facilitating the processing of live data streams and providing insightful analytics instantaneously.

For batch processing, I am using  Dask, Spark, and Pandas to crunch large volumes of data and serving layer with MongoDB, providing a resilient and flexible storage solution that adapts predicitive maintenance requirements.

For orchestration pipelines, the repo is using airflow, where the run frequency can be set based on solution requirements

API layer provided by FastAPI, allowing easy and fast interaction and integration with your existing systems.
