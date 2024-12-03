# Module 07 Assignment 01: Programming Assignment 2

## Author Information
- **Name**: Advaithbarath Raghuraman Bhuvaneswari
- **UCID**: ar2728
- **Docker**: [Your Docker Hub Link]

---

## Assignment Summary
This assignment involves building a machine learning application for predicting wine quality using Apache Spark on the Amazon AWS cloud platform. The goal is to train a model in parallel across multiple EC2 instances and deploy it effectively. The application uses provided datasets to train and validate a Spark MLlib model in parallel on four EC2 instances. The trained model is then deployed on a single EC2 instance for predictions, with its performance measured using the F1 score. Additionally, the prediction application is containerized using Docker for efficient deployment. The implementation is completed in Java on Ubuntu Linux, utilizing parallel computing, machine learning model development, and containerization.

---

## Steps for AWS Credential Setup

### Via AWS Academy Learner Lab (New User)
1. Create an AWS account using your NJIT email.
2. Navigate to **Academy Learner Lab > Modules > Launch Learner Academy Lab**.
3. Open the Learner Lab environment.

### Via Vocareum (Existing User)
1. Log in with your NJIT email.
2. Select **Student Role** and open the lab link.
3. Access the Learner Lab environment.

---

## Launching AWS Management Console
1. Start the lab and wait until the AWS symbol turns green.
2. Click the AWS symbol to open the AWS Management Console (valid for 4 hours).

---

## Creating EMR Cluster
1. Search for "EMR" in the AWS Management Console.
2. Create or clone a cluster with the following settings:
   - Name: `Wine_Cluster`
   - Instance type: `m4.large` (for all instances).
   - Cluster termination: Set to **manual**.
   - Private key: `cluster.ppk`.

3. Wait for the cluster to initialize.

---

## Setting SSH for Master EC2 Instance
1. Search for EC2 and go to the **Security Groups** section.
2. Select the security group for the ElasticMapReduce master.
3. Edit inbound rules to ensure SSH access is enabled.
4. Connect to the EC2 master instance via SSM.

---

## Setting Up S3 Bucket
1. Create an S3 bucket (e.g., `bucketcluster`).
2. Upload the following files:
   - `training.py`
   - `TrainingDataset.csv`
   - `prediction.py`
   - `ValidationDataset.csv`
3. After executing `training.py`, a `trainedmodel` folder will be created in the bucket.

---

## Execution Without Docker

### Steps:
1. Access the EC2 instance and install dependencies:
>>> pip install numpy

2. Execute the training script:
>>> spark-submit s3://bucketcluster/training.py

3. Execute the prediction script:
>>> spark-submit s3://bucketcluster/prediction.py s3://bucketcluster/ValidationDataset.csv


### Results:
- Accuracy: 96%
- F1 Score: 0.954791

---

## Execution With Docker

### Steps:
1. Build and push the Docker image:
- Build:

>>> docker build -t advaith123/bucketcluster:predict .

- Tag and push:

>>>  docker tag advaith123/bucketcluster:predict advaith123/bucketcluster:predict
>>>  docker push advaith123/bucketcluster:predict

2. Start Docker on the EC2 instance:
>>> sudo systemctl start docker sudo systemctl enable docker

3. Pull the Docker image:
>>> sudo docker pull advaith123/bucketcluster:predict

4. Run the prediction script using Docker:
>>> sudo docker run -v /home/ec2-user/:/job advaith123/bucketcluster:predict


### Results:
- Accuracy: 96%
- F1 Score: 0.954791

---

## Notes and Best Practices

### AWS Sessions
- AWS Management Console sessions last for 4 hours. Ensure all necessary tasks are completed within this time or relaunch the lab.

### EMR Cluster
- EMR clusters terminate automatically after the session ends unless the termination setting is set to manual.
- Clusters can be cloned for reuse to save setup time.

### S3 Bucket
- Ensure all required files are uploaded to the S3 bucket before starting the training or prediction process.
- The `trainedmodel` folder will be created in the S3 bucket upon successful training.

### Docker Usage
- Docker simplifies deployment and ensures consistent runtime environments.
- Always verify the Docker image version and test it locally before pulling it to an EC2 instance.

---

## Expected Outcomes
- A trained machine learning model for wine quality prediction.
- High prediction accuracy (96%) and F1 score (0.954791).
- A containerized application for efficient deployment across environments.

---

## References
- Apache Spark MLlib Documentation
- AWS EMR User Guide
- Docker Official Documentation

---

## Contact
- **Author**: Advaithbarath Raghuraman Bhuvaneswari
- **Email**: [Your Email]
- **GitHub**: [Your GitHub Profile]
- **Docker**: [Your Docker Hub Profile]
