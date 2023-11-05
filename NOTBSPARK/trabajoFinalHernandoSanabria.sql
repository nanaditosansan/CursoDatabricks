-- Databricks notebook source


-- COMMAND ----------

-- MAGIC %md
-- MAGIC # U.A.G.R.M

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # SCHOOL OF ENGINEERING - UNIDAD DE POSTGRADO

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## MAESTRIA EN CIENCIA DE DATOS Y BIG DATA

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### MODULO 14 TECNOLOGÍAS PARA BIG DATA

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### DOCENTE: MSc. Danny Huanca Sevilla

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Trabajo Final Aplicación Spark Machine Learning

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Postgraduante: Hernando Sanabria Yupanqui

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Importando librerias

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml.classification import LogisticRegression

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml import Pipeline

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml.classification import RandomForestClassifier

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml.classification import NaiveBayes

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql import functions as F

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC import matplotlib.pyplot as plt

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pyspark
-- MAGIC print(pyspark.__version__)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Comprensión de los datos

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Recopilación de los datos

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Obteniendo los datos ingestados desde el DBMS y creandolos en una tabla

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_ACCOUNT;

CREATE TABLE BERKA_ACCOUNT
USING csv
OPTIONS (path "/FileStore/BERKA2/account.asc", header "true")


-- COMMAND ----------

select * from BERKA_ACCOUNT

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_CARD;

CREATE TABLE BERKA_CARD
USING csv
OPTIONS (path "/FileStore/BERKA2/card.asc", header "true")


-- COMMAND ----------

select * from BERKA_CARD

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_CLIENT;

CREATE TABLE BERKA_CLIENT
USING csv
OPTIONS (path "/FileStore/BERKA2/client.asc", header "true")

-- COMMAND ----------

select * from BERKA_CLIENT

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_DISP;

CREATE TABLE BERKA_DISP
USING csv
OPTIONS (path "/FileStore/BERKA2/disp.asc", header "true")

-- COMMAND ----------

select * from BERKA_DISP

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_DISTRICT;

CREATE TABLE BERKA_DISTRICT
USING csv
OPTIONS (path "/FileStore/BERKA2/district.asc", header "true")

-- COMMAND ----------

select * from BERKA_DISTRICT

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_LOAN;

CREATE TABLE BERKA_LOAN
USING csv
OPTIONS (path "/FileStore/BERKA2/loan.asc", header "true")

-- COMMAND ----------

select * from BERKA_LOAN

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_ORDER;

CREATE TABLE BERKA_ORDER
USING csv
OPTIONS (path "/FileStore/BERKA2/order.asc", header "true")

-- COMMAND ----------

select * from BERKA_ORDER

-- COMMAND ----------

DROP TABLE IF EXISTS BERKA_TRANS;

CREATE TABLE BERKA_TRANS
USING csv
OPTIONS (path "/FileStore/BERKA2/trans.asc", header "true")

-- COMMAND ----------

select * from BERKA_TRANS

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Descripción de los datos

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Vemos que cantidad de transacciones tenemos

-- COMMAND ----------

select count(*) from BERKA_TRANS

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Vemos el schema de la tabla transaccional

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfTRans=spark.sql("select * from BERKA_TRANS")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfTRans.printSchema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC El campo amount debiera ser numérico
-- MAGIC
-- MAGIC El campo date debiera ser tipo fecha

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Veamos la tabla loan o préstamos donde esta la clasificación de las cuentas A B C o D

-- COMMAND ----------

select count(*) from BERKA_LOAN

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoan=spark.sql("select * from BERKA_LOAN")
-- MAGIC
-- MAGIC dfLoan.printSchema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC El campo duration que es la duración del prestámo  debiera ser numérico.
-- MAGIC
-- MAGIC Un atributo relevante es el campo status de la tabla loan pues este determina en que estado está una cuenta A B C o D.
-- MAGIC
-- MAGIC Este es un campo importante pues  ayudará a determinar el resultado o variable objetivo del modelo el estado de cuenta A B C o D.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_loan_pandas = dfLoan[['status','loan_id']].groupBy(F.col('status')).count().toPandas()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Obteniendo un gráfico de barras de las categorias de prestamos

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_loan_pandas

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_loan_pandas.plot.bar('status', figsize=(10,5))
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Podemos ver de esta variable que la gran mayoría de las cuentas están en el tipo C es decir no tienen problemas.
-- MAGIC
-- MAGIC Los de tipo B son los más sensibles pues estas deudas no se habrían pagado pero son la minoría.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Exploración de los datos

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoan=dfLoan.withColumn("AMOUNT", col("amount").cast("long"))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoan.printSchema()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_loan_pandas1 = dfLoan[['AMOUNT']].toPandas()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Vemos que el monto de los prestamos estan sesgados a la derecha
-- MAGIC
-- MAGIC Se puede observar que la gran mayoría de los préstamos están entre 50000 y 150000 en la moneda del país del que es el banco.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_loan_pandas1['AMOUNT'].hist(bins=10)
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoanTipoB=dfLoan[dfLoan['status']=='B']
-- MAGIC dfLoanTipoB

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoanTipoBPandas=dfLoanTipoB.toPandas()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoanTipoBPandas

-- COMMAND ----------

-- MAGIC %md
-- MAGIC También se hizo un histograma de los préstamos tipo “B” que es caso mas critico en los que no se habria pagado la deuda. Podemos observar que los montos de las cuentas están entre 0 y 50000. Además existe un dato atípico con montos mayores a 400000.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoanTipoBPandas['AMOUNT'].hist(bins=10)
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Obtenemos un describe de los montos de los prestamos críticos tipo B

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dfLoanTipoBPandas.describe()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Formulamos la hipótesis que la clasificación del tipo de cuenta depende de la combinación de la cantidad de meses en las que se deberia pagar un préstamo, la cantidad de meses efectivamente pagados y la cantidad de meses que deberia de tener pagado a la fecha.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Verificación de la calidad

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Verificamos que los campos de la tabla que clasifica las cuentas no tenga nulos

-- COMMAND ----------

select * from BERKA_LOAN where status is null

-- COMMAND ----------

select * from BERKA_LOAN where duration is null

-- COMMAND ----------

select * from BERKA_LOAN where date is null

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Preparación de los datos

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Realizaremos integración de los datos partiendo de la tabla transaccional haciendo joins con sus tablas relacionadas como por ejemplo loan.
-- MAGIC
-- MAGIC Se hace una traducción de varios tipos de registros de su valor reducido a una descripción mas detallada
-- MAGIC
-- MAGIC Se agrupa las transacciones de acuerdo a la tabla con la que se relacionan por ejemplo las transacciones relacionadas con la tabla order se marcan como tipo O.

-- COMMAND ----------

SELECT A.ACCOUNT_ID ,O.AMOUNT, 'O' TIPO, NULL FECHA,'' TIPOT, '' SUMARESTA ,'' OPERATION,'' OPERATIONTRA, 0 BALANCE,O.K_SYMBOL  TKSYMBOLORI,
CASE O.K_SYMBOL WHEN 'POJISTNE' THEN 'PAGO DE SEGURO' WHEN 'SIPO' THEN 'PAGO FAMILIAR' WHEN 'LEASING' THEN 'ALQUILER'  WHEN 'UVER' THEN 'PAGO PRESTAMO' END TKSYMBOL,
A.date  FECHACREACUENTA,
A.FREQUENCY ,
CASE A.FREQUENCY WHEN 'POPLATEK MESICNE' THEN 'EMISION MENSUAL' WHEN 'POPLATEK TYDNE' THEN 'EMISION SEMANAL' WHEN 'POPLATEK PO OBRATU' THEN 'EMSION DESPUES TRANSACCION' END FRECUENCYSIG
FROM BERKA_ACCOUNT  A, BERKA_ORDER  O 
WHERE A.ACCOUNT_ID =O.ACCOUNT_ID 

-- COMMAND ----------

SELECT A.ACCOUNT_ID ,T.AMOUNT ,'T' TIPO,T.date FECHA  , T.TYPE TIPOT,
CASE T.TYPE WHEN 'PRIJEM' THEN 'MAS' WHEN 'VYDAJ' THEN 'MENOS' END SUMARESTA ,
T.OPERATION ,
CASE T.OPERATION  WHEN 'VYBER KARTOU' THEN 'TARJETA CREDITO RETIRO' WHEN 'VKLAD' THEN 'CREDITO EN EFECTIVO' WHEN 'PREVOD Z UCTU' THEN 'RECAUDACION OTRO BANCO' WHEN 'VYBER' THEN 'RETIRO EFECTIVO' WHEN 'PREVOD NA UCET' THEN 'REMESAS A OTRO BANCO' END OPERATIONTRA,
T.BALANCE,T.K_SYMBOL TKSYMBOLORI,
CASE T.K_SYMBOL  WHEN 'POJISTNE' THEN 'PAGO DE SEGUROS' WHEN 'SLUZBY' THEN 'PAGO ESTADO CUENTA' WHEN 'UROK' THEN 'INTERES ACREDITADO' WHEN 'SANKC. UROK' THEN 'INTERES SANCIONADOR SI BALANCE NEGATIVO' WHEN 'SIPO' THEN 'HOGAR' WHEN 'DUCHOD' THEN 'PENSION VEJEZ' WHEN 'UVER' THEN 'PAGO DE PRESTAMO' END TKSYMBOL,
A.date  FECHACREACUENTA,
A.FREQUENCY ,
CASE A.FREQUENCY WHEN 'POPLATEK MESICNE' THEN 'EMISION MENSUAL' WHEN 'POPLATEK TYDNE' THEN 'EMISION SEMANAL' WHEN 'POPLATEK PO OBRATU' THEN 'EMSION DESPUES TRANSACCION' END FRECUENCYSIG
FROM BERKA_ACCOUNT A, BERKA_TRANS T
WHERE A.ACCOUNT_ID =T.ACCOUNT_ID 

-- COMMAND ----------

SELECT l.ACCOUNT_ID,l.AMOUNT , 'L' TIPO, l.date FECHA, '' TIPOT,
'' SUMARESTA, '' OPERATION, '' OPERATIONTRA, 0 BALANCE, l.STATUS  TKSYMBOLORI,
CASE l.STATUS WHEN 'A' THEN 'CONTRATO TERMINADO NO PROBLEMAS' WHEN 'B' THEN 'CONTRATO TERMINADO PRESTAMO NO PAGADO' WHEN 'C' THEN 'CONTRATO VIGENTE MUY LEJOS' WHEN 'D' THEN 'CONTRATO VIGENTE CLIENTE EN DEUDA' END   TKSYMBOL,
A.date  FECHACREACUENTA,
A.FREQUENCY ,
CASE A.FREQUENCY WHEN 'POPLATEK MESICNE' THEN 'EMISION MENSUAL' WHEN 'POPLATEK TYDNE' THEN 'EMISION SEMANAL' WHEN 'POPLATEK PO OBRATU' THEN 'EMSION DESPUES TRANSACCION' END FRECUENCYSIG
FROM BERKA_LOAN l, BERKA_ACCOUNT  a 
WHERE l.ACCOUNT_ID =a.ACCOUNT_ID 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Se genera la union de todos los tipos de transacciones en una sola tabla llamada TABLAMINABLE1

-- COMMAND ----------

DROP TABLE IF EXISTS TMINABLE1;
CREATE TABLE TMINABLE1 AS
SELECT * FROM (
SELECT A.ACCOUNT_ID ,O.AMOUNT, 'O' TIPO, NULL FECHA,'' TIPOT, '' SUMARESTA ,'' OPERATION,'' OPERATIONTRA, 0 BALANCE,O.K_SYMBOL  TKSYMBOLORI,
CASE O.K_SYMBOL WHEN 'POJISTNE' THEN 'PAGO DE SEGURO' WHEN 'SIPO' THEN 'PAGO FAMILIAR' WHEN 'LEASING' THEN 'ALQUILER'  WHEN 'UVER' THEN 'PAGO PRESTAMO' END TKSYMBOL,
A.date  FECHACREACUENTA,
A.FREQUENCY ,
CASE A.FREQUENCY WHEN 'POPLATEK MESICNE' THEN 'EMISION MENSUAL' WHEN 'POPLATEK TYDNE' THEN 'EMISION SEMANAL' WHEN 'POPLATEK PO OBRATU' THEN 'EMSION DESPUES TRANSACCION' END FRECUENCYSIG
FROM BERKA_ACCOUNT  A, BERKA_ORDER  O 
WHERE A.ACCOUNT_ID =O.ACCOUNT_ID 

UNION ALL

SELECT A.ACCOUNT_ID ,T.AMOUNT ,'T' TIPO,T.date FECHA  , T.TYPE TIPOT,
CASE T.TYPE WHEN 'PRIJEM' THEN 'MAS' WHEN 'VYDAJ' THEN 'MENOS' END SUMARESTA ,
T.OPERATION ,
CASE T.OPERATION  WHEN 'VYBER KARTOU' THEN 'TARJETA CREDITO RETIRO' WHEN 'VKLAD' THEN 'CREDITO EN EFECTIVO' WHEN 'PREVOD Z UCTU' THEN 'RECAUDACION OTRO BANCO' WHEN 'VYBER' THEN 'RETIRO EFECTIVO' WHEN 'PREVOD NA UCET' THEN 'REMESAS A OTRO BANCO' END OPERATIONTRA,
T.BALANCE,T.K_SYMBOL TKSYMBOLORI,
CASE T.K_SYMBOL  WHEN 'POJISTNE' THEN 'PAGO DE SEGUROS' WHEN 'SLUZBY' THEN 'PAGO ESTADO CUENTA' WHEN 'UROK' THEN 'INTERES ACREDITADO' WHEN 'SANKC. UROK' THEN 'INTERES SANCIONADOR SI BALANCE NEGATIVO' WHEN 'SIPO' THEN 'HOGAR' WHEN 'DUCHOD' THEN 'PENSION VEJEZ' WHEN 'UVER' THEN 'PAGO DE PRESTAMO' END TKSYMBOL,
A.date  FECHACREACUENTA,
A.FREQUENCY ,
CASE A.FREQUENCY WHEN 'POPLATEK MESICNE' THEN 'EMISION MENSUAL' WHEN 'POPLATEK TYDNE' THEN 'EMISION SEMANAL' WHEN 'POPLATEK PO OBRATU' THEN 'EMSION DESPUES TRANSACCION' END FRECUENCYSIG
FROM BERKA_ACCOUNT A, BERKA_TRANS T
WHERE A.ACCOUNT_ID =T.ACCOUNT_ID 

UNION ALL

SELECT l.ACCOUNT_ID,l.AMOUNT , 'L' TIPO, l.date FECHA, '' TIPOT,
'' SUMARESTA, '' OPERATION, '' OPERATIONTRA, 0 BALANCE, l.STATUS  TKSYMBOLORI,
CASE l.STATUS WHEN 'A' THEN 'CONTRATO TERMINADO NO PROBLEMAS' WHEN 'B' THEN 'CONTRATO TERMINADO PRESTAMO NO PAGADO' WHEN 'C' THEN 'CONTRATO VIGENTE MUY LEJOS' WHEN 'D' THEN 'CONTRATO VIGENTE CLIENTE EN DEUDA' END   TKSYMBOL,
A.date  FECHACREACUENTA,
A.FREQUENCY ,
CASE A.FREQUENCY WHEN 'POPLATEK MESICNE' THEN 'EMISION MENSUAL' WHEN 'POPLATEK TYDNE' THEN 'EMISION SEMANAL' WHEN 'POPLATEK PO OBRATU' THEN 'EMSION DESPUES TRANSACCION' END FRECUENCYSIG
FROM BERKA_LOAN l, BERKA_ACCOUNT  a 
WHERE l.ACCOUNT_ID =a.ACCOUNT_ID 
) t1
ORDER BY t1.ACCOUNT_ID,t1.FECHA


-- COMMAND ----------

select * from TMINABLE1

-- COMMAND ----------

select count(*) from TMINABLE1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Empezamos a construir nuevos datos agregando los datos de la anterior tabla por ejemplo generamos cuantos meses se debe pagar por cada cuenta en la tabla CUANROSMESESDEBEPAGAR.
-- MAGIC
-- MAGIC Cuantos meses ya se han pagado de una cuenta en la tabla MESESPAGADOS
-- MAGIC
-- MAGIC Y cuántos meses deberia ya tener pagado a la fecha tomando en cuenta la fecha de inicio del prestamo en la tabla CANTIDADMESESDEBERIAPAGAR
-- MAGIC

-- COMMAND ----------

DROP TABLE IF EXISTS CUANROSMESESDEBEPAGAR;
CREATE TABLE CUANROSMESESDEBEPAGAR   AS	
SELECT t1.ACCOUNT_ID, L.DURATION  FROM BERKA_LOAN L, TMINABLE1 t1
WHERE L.ACCOUNT_ID =t1.ACCOUNT_ID
AND t1.TIPO='L'


-- COMMAND ----------

select * from CUANROSMESESDEBEPAGAR

-- COMMAND ----------

DROP TABLE IF EXISTS MESESPAGADOS;
CREATE TABLE MESESPAGADOS  AS  
SELECT ACCOUNT_ID, count(ACCOUNT_ID) MESESPAGADOS FROM TMINABLE1 WHERE tipo='T' AND TKSYMBOLORI='UVER'
GROUP BY ACCOUNT_ID
ORDER BY ACCOUNT_ID

-- COMMAND ----------

select * from MESESPAGADOS

-- COMMAND ----------

--SELECT ACCOUNT_ID ,FECHA,
--(12-EXTRACT(MONTH FROM to_date(FECHA,'yymmdd')+1)+(1998-EXTRACT(YEAR FROM to_date(FECHA,#'ddmmdd'))*12 TOTALMESES
--FROM TMINABLE1 WHERE tipo='L'
--AND  EXTRACT(DAY FROM FECHA)<=12


-- COMMAND ----------

--SELECT to_date(FECHA,'yyMMdd') FROM TMINABLE1

-- COMMAND ----------

DROP TABLE IF EXISTS CANTIDADMESESDEBERIAPAGAR;
CREATE TABLE CANTIDADMESESDEBERIAPAGAR
USING csv
OPTIONS (path "/FileStore/BERKA2/CANTIDADMESESDEBERIAPAGAR.csv", header "true")

-- COMMAND ----------

--CREATE TABLE CANTIDADMESESDEBERIAPAGAR  AS  
--SELECT t.* FROM(
--SELECT ACCOUNT_ID ,FECHA,
--(12-EXTRACT(MONTH FROM FECHA)+1)+(1998-EXTRACT(YEAR FROM FECHA))*12 TOTALMESES
--FROM TMINABLE1 WHERE tipo='L'
--AND  EXTRACT(DAY FROM FECHA)<=12

--UNION ALL

--SELECT ACCOUNT_ID ,FECHA,
--((12-EXTRACT(MONTH FROM FECHA)+1)+(1998-EXTRACT(YEAR FROM FECHA))*12)-1 TOTALMESES
--FROM TMINABLE1 WHERE tipo='L'
--AND  EXTRACT(DAY FROM FECHA)>12
--)t
--ORDER BY t.ACCOUNT_ID


-- COMMAND ----------



-- COMMAND ----------

select * from CANTIDADMESESDEBERIAPAGAR

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Generamos una primera versión de la tabla minable con la combinación de los anteriores datos en la tabla TMINABLE4

-- COMMAND ----------

DROP TABLE IF EXISTS TMINABLE4;
CREATE TABLE TMINABLE4   AS  
SELECT c1.ACCOUNT_ID,c4.AMOUNT ,DURATION MESESDEBEPAGAR,c2.MESESPAGADOS,c3.TOTALMESES MESESDEBERIAPAGAR,c4.TKSYMBOLORI 
FROM CUANROSMESESDEBEPAGAR c1, MESESPAGADOS c2,CANTIDADMESESDEBERIAPAGAR c3,TMINABLE1 c4
WHERE c1.ACCOUNT_ID=c2.ACCOUNT_ID
AND c1.ACCOUNT_ID=c3.ACCOUNT_ID
AND c1.ACCOUNT_ID=c4.ACCOUNT_ID
AND c4.TIPO ='L'


-- COMMAND ----------

select * from TMINABLE4

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %md
-- MAGIC Como la variable  objetivo no debe ser expresado en forma de letras para que funcionen los modelos se cambia el mismo de este tipo a  numérico

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df1=spark.sql("select * from TMINABLE4")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df1.show(20)

-- COMMAND ----------



-- COMMAND ----------

select distinct(TKSYMBOLORI) from TMINABLE4

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Convertiremos la columna objetivo con index

-- COMMAND ----------

-- MAGIC %python
-- MAGIC string_indexer = StringIndexer(inputCol="TKSYMBOLORI", outputCol="IndexTKSYMBOLORI")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC model = string_indexer.fit(df1)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df2 = model.transform(df1)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df2.show(20)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df1.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df3=df2.drop("TKSYMBOLORI")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df3.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Como las columnas ACCOUNT_ID y AMOUNT no contribuiran al modelo se excluyen de la tabla minable

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4=df2.drop("ACCOUNT_ID")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4=df4.drop("AMOUNT")
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4=df4.drop("TKSYMBOLORI")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4.printSchema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Cambiando el schema de la tabla minable para que todo sea numérico

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4=df4.withColumn("MESESDEBEPAGAR", col("MESESDEBEPAGAR").cast("int"))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4=df4.withColumn("MESESPAGADOS", col("MESESPAGADOS").cast("int"))
-- MAGIC df4=df4.withColumn("MESESDEBERIAPAGAR", col("MESESDEBERIAPAGAR").cast("int"))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4.printSchema()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df4.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Modelado

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Ahora aplicaremos regresion lineal multiclase para aplicar a la tabla minable

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %python
-- MAGIC assembler = VectorAssembler(inputCols = [
-- MAGIC                                         'MESESDEBEPAGAR',
-- MAGIC                                          'MESESPAGADOS',
-- MAGIC                                          'MESESDEBERIAPAGAR'
-- MAGIC                                         ], outputCol= 'features')

-- COMMAND ----------

-- MAGIC %python
-- MAGIC lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial", labelCol="IndexTKSYMBOLORI")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC pipeline = Pipeline(stages=[assembler, lr])

-- COMMAND ----------

-- MAGIC %python
-- MAGIC (trainingData, testData) = df4.randomSplit([0.7, 0.3])

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %python
-- MAGIC model = pipeline.fit(trainingData)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC predictions = model.transform(testData)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC evaluator = MulticlassClassificationEvaluator(labelCol="IndexTKSYMBOLORI", predictionCol="prediction", metricName="accuracy")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC accuracy = evaluator.evaluate(predictions)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print("Accuracy = %g" % (accuracy))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Ahora probamos con otro modelo como es Random Forest

-- COMMAND ----------

-- MAGIC %python
-- MAGIC assembler = VectorAssembler(inputCols = [
-- MAGIC                                         'MESESDEBEPAGAR',
-- MAGIC                                          'MESESPAGADOS',
-- MAGIC                                          'MESESDEBERIAPAGAR'
-- MAGIC                                         ], outputCol= 'features')

-- COMMAND ----------

-- MAGIC %python
-- MAGIC rf = RandomForestClassifier(numTrees=100, labelCol="IndexTKSYMBOLORI", featuresCol="features")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC (trainingData, testData) = df4.randomSplit([0.7, 0.3])

-- COMMAND ----------

-- MAGIC %python
-- MAGIC pipeline = Pipeline(stages=[assembler, rf])

-- COMMAND ----------

-- MAGIC %python
-- MAGIC model = pipeline.fit(trainingData)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC predictions = model.transform(testData)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC evaluator = MulticlassClassificationEvaluator(labelCol="IndexTKSYMBOLORI", predictionCol="prediction", metricName="accuracy")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC accuracy = evaluator.evaluate(predictions)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print("Accuracy = %g" % (accuracy))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Probamos el algoritmo Naive Bayes

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %python
-- MAGIC assembler = VectorAssembler(inputCols = [
-- MAGIC                                         'MESESDEBEPAGAR',
-- MAGIC                                          'MESESPAGADOS',
-- MAGIC                                          'MESESDEBERIAPAGAR'
-- MAGIC                                         ], outputCol= 'features')

-- COMMAND ----------

-- MAGIC %python
-- MAGIC nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="IndexTKSYMBOLORI", featuresCol="features")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC (trainingData, testData) = df4.randomSplit([0.7, 0.3])

-- COMMAND ----------

-- MAGIC %python
-- MAGIC pipeline = Pipeline(stages=[assembler, nb])

-- COMMAND ----------

-- MAGIC %python
-- MAGIC model = pipeline.fit(trainingData)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC predictions = model.transform(testData)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC evaluator = MulticlassClassificationEvaluator(labelCol="IndexTKSYMBOLORI", predictionCol="prediction", metricName="accuracy")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC accuracy = evaluator.evaluate(predictions)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print("Accuracy = %g" % (accuracy))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Podemos observar que el mejor modelo es el de Random Forest con un 92% de Acuraccy

-- COMMAND ----------


