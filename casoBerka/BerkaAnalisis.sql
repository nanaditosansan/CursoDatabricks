-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Analisis de Churn de la empresa BERKA

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

CREATE TABLE CUANROSMESESDEBEPAGAR   AS	
SELECT t1.ACCOUNT_ID, L.DURATION  FROM BERKA_LOAN L, TMINABLE1 t1
WHERE L.ACCOUNT_ID =t1.ACCOUNT_ID
AND t1.TIPO='L'


-- COMMAND ----------

select * from CUANROSMESESDEBEPAGAR

-- COMMAND ----------

CREATE TABLE MESESPAGADOS  AS  
SELECT ACCOUNT_ID, count(ACCOUNT_ID) MESESPAGADOS FROM TMINABLE1 WHERE tipo='T' AND TKSYMBOLORI='UVER'
GROUP BY ACCOUNT_ID
ORDER BY ACCOUNT_ID

-- COMMAND ----------

select * from MESESPAGADOS

-- COMMAND ----------

SELECT ACCOUNT_ID ,FECHA,
(12-EXTRACT(MONTH FROM to_date(FECHA,'yymmdd')+1)+(1998-EXTRACT(YEAR FROM to_date(FECHA,'ddmmdd'))*12 TOTALMESES
FROM TMINABLE1 WHERE tipo='L'
AND  EXTRACT(DAY FROM FECHA)<=12


-- COMMAND ----------

SELECT to_date(FECHA,'yyMMdd') FROM TMINABLE1

-- COMMAND ----------

CREATE TABLE CANTIDADMESESDEBERIAPAGAR
USING csv
OPTIONS (path "/FileStore/BERKA2/CANTIDADMESESDEBERIAPAGAR.csv", header "true")

-- COMMAND ----------

CREATE TABLE CANTIDADMESESDEBERIAPAGAR  AS  
SELECT t.* FROM(
SELECT ACCOUNT_ID ,FECHA,
(12-EXTRACT(MONTH FROM FECHA)+1)+(1998-EXTRACT(YEAR FROM FECHA))*12 TOTALMESES
FROM TMINABLE1 WHERE tipo='L'
AND  EXTRACT(DAY FROM FECHA)<=12

UNION ALL

SELECT ACCOUNT_ID ,FECHA,
((12-EXTRACT(MONTH FROM FECHA)+1)+(1998-EXTRACT(YEAR FROM FECHA))*12)-1 TOTALMESES
FROM TMINABLE1 WHERE tipo='L'
AND  EXTRACT(DAY FROM FECHA)>12
)t
ORDER BY t.ACCOUNT_ID


-- COMMAND ----------



-- COMMAND ----------

select * from CANTIDADMESESDEBERIAPAGAR

-- COMMAND ----------

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



-- COMMAND ----------

