from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import regexp_replace, col
import time

# Initialize Spark session
spark = SparkSession.builder.appName("IMDB_Movie_Analysis").getOrCreate()

# Load dataset
df = spark.read.csv("hdfs://192.168.13.89:9000/sat5165/movies.csv", header=True, inferSchema=True)

# Data Preprocessing
df = df.withColumn("Total_Gross", regexp_replace("Total_Gross", "[$M,]", "").cast("float"))
df = df.withColumn("main_genre", regexp_replace("main_genre", " ", ""))
df = df.withColumn("side_genre", regexp_replace("side_genre", " ", ""))
df = df.filter((df["Total_Gross"].isNotNull()) & (df["Total_Gross"] > 0))

# Define a binary target (success is if Total_Gross > 100 million)
df = df.withColumn("successful", (col("Total_Gross") > 100).cast("int"))

# Encode 'main_genre' as a numerical feature
indexer = StringIndexer(inputCol="main_genre", outputCol="main_genre_index")
df = indexer.fit(df).transform(df)

# Assemble feature vector
assembler = VectorAssembler(inputCols=["Rating", "Runtime(Mins)", "main_genre_index"], outputCol="features")
assembled_data = assembler.transform(df)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaled_data = scaler.fit(assembled_data).transform(assembled_data)

# Split data
train_data, test_data = scaled_data.randomSplit([0.8, 0.2], seed=42)

# Linear Regression Model
lr = LinearRegression(featuresCol="scaled_features", labelCol="Total_Gross")
start_time = time.time()
lr_model = lr.fit(train_data)
lr_time = time.time() - start_time
lr_predictions = lr_model.transform(test_data)

# Linear Regression Evaluation
reg_evaluator = RegressionEvaluator(labelCol="Total_Gross", predictionCol="prediction", metricName="r2")
r2 = reg_evaluator.evaluate(lr_predictions)

# Save LR results
with open("/home/sat3812/Documents/linear_regression_results.txt", "w") as f:
	f.write(f"Linear Regression R2 Score: {r2}\n")
	f.write(f"Training Time for Linear Regression: {lr_time} seconds\n")

# Random Forest Model for "Successful" movies
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="successful")
start_time = time.time()
rf_model = rf.fit(train_data)
rf_time = time.time() - start_time
rf_predictions = rf_model.transform(test_data)

# Random Forest Evaluation
classifier_evaluator = BinaryClassificationEvaluator(labelCol="successful", rawPredictionCol="prediction", metricName="areaUnderROC")
auc = classifier_evaluator.evaluate(rf_predictions)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="successful", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="successful", predictionCol="prediction", metricName="weightedRecall")

precision = precision_evaluator.evaluate(rf_predictions)
recall = recall_evaluator.evaluate(rf_predictions)

# Save RF results
with open("/home/sat3812/Documents/random_forest_results.txt", "w") as f:
	f.write(f"Random Forest AUC Score: {auc}\n")
	f.write(f"Random Forest Precision: {precision}\n")
	f.write(f"Random Forest Recall: {recall}\n")
	f.write(f"Training Time for Random Forest: {rf_time} seconds\n")

