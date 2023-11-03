# Databricks notebook source
# MAGIC %md
# MAGIC # U.A.G.R.M

# COMMAND ----------

# MAGIC %md
# MAGIC # SCHOOL OF ENGINEERING - UNIDAD DE POSTGRADO

# COMMAND ----------

# MAGIC %md
# MAGIC ## MAESTRIA EN CIENCIA DE DATOS Y BIG DATA

# COMMAND ----------

# MAGIC %md
# MAGIC ### MODULO 14 TECNOLOGÍAS PARA BIG DATA

# COMMAND ----------

# MAGIC %md
# MAGIC ### DOCENTE: MSc. Danny Huanca Sevilla

# COMMAND ----------

# MAGIC %md
# MAGIC ##Laboratorio Spark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC ### Postgraduante: Hernando Sanabria Yupanqui

# COMMAND ----------

# MAGIC %md
# MAGIC FECHA 05/11/23

# COMMAND ----------

 employee_df=spark.read.json("dbfs:/FileStore/PRASPARK/employees.json")

# COMMAND ----------

employee_df.show()

# COMMAND ----------

employee_df.printSchema()

# COMMAND ----------

employee_df.select("designation").show()

# COMMAND ----------

employee_df.select("sal").show()

# COMMAND ----------

employee_df.select("*").show()

# COMMAND ----------

employee_df.select("deptno", "designation", "ename").show()

# COMMAND ----------

employee_df.select(employee_df['sal']+10).show()

# COMMAND ----------

employee_df.filter(employee_df['sal'] > 2000).show()

# COMMAND ----------

employee_df.groupBy(employee_df['designation']).count().show()

# COMMAND ----------

employee_df.createOrReplaceTempView("empleados")

# COMMAND ----------

sqlDF = spark.sql("select * from empleados")
sqlDF.show()

# COMMAND ----------

employee_df.createGlobalTempView("empleadosGlobal2")

# COMMAND ----------

spark.sql("select * from global_temp.empleadosGlobal2").show()

# COMMAND ----------

spark.sql("select count(*) from empleados where ename = 'SMITH'").show()

# COMMAND ----------

employee_df.createOrReplaceTempView("empleados2")

# COMMAND ----------

spark.sql("select * from empleados a \
           left join empleados2 b on a.deptno = b.deptno").show()

# COMMAND ----------

# se puee realizar tambien fucniones analiticas

spark.sql("select deptno, designation, round(avg(sal) over (partition by deptno),2) mediadeptno from empleados").show()

# COMMAND ----------

# ¿Cuántos empleados trabajan en cada uno de los deptno?
spark.sql("select deptno,count(empno) nroEmpleados from empleados group by deptno").show()

# COMMAND ----------

# Determine cual es el promedio de salario de todos los empleados
spark.sql("select avg(sal) promedio from empleados").show()

# COMMAND ----------

# Realice un join entre la tabla empleados y empleados2 y determine el promedio de salario por deptno
spark.sql("select a.deptno, avg(a.sal) promedio from empleados a, empleados2 b group by a.deptno").show()

# COMMAND ----------

# Cree una funcion analitica para ordenar por salario de menor a mayor para cada departamento
spark.sql("select deptno,sal from empleados order by deptno, sal asc").show()

# COMMAND ----------

pandas_df = employee_df.toPandas()

# COMMAND ----------

pandas_df.head()

# COMMAND ----------


